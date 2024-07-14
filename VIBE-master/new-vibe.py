#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 13:19:31 2019
升级版vibe（速度更快）
@author: youxinlin
"""

import numpy as np
import cv2
import detect_api
import torch


class ViBe:
    '''
    ViBe运动检测，分割背景和前景运动图像
    '''

    def __init__(self, num_sam=20, min_match=2, radiu=20, rand_sam=16):
        self.defaultNbSamples = num_sam  # 每个像素的样本集数量，默认20个
        self.defaultReqMatches = min_match  # 前景像素匹配数量，如果超过此值，则认为是背景像素
        self.defaultRadius = radiu  # 匹配半径，即在该半径内则认为是匹配像素
        self.defaultSubsamplingFactor = rand_sam  # 随机数因子，如果检测为背景，每个像素有1/defaultSubsamplingFactor几率更新样本集和领域样本集

        self.background = 0
        self.foreground = 255

    def __buildNeighborArray(self, img):
        '''
        构建一副图像中每个像素的邻域数组
        参数：输入灰度图像
        返回值：每个像素9邻域数组，保存到self.samples中
        '''
        height, width = img.shape
        self.samples = np.zeros((self.defaultNbSamples, height, width), dtype=np.uint8)

        # 生成随机偏移数组，用于计算随机选择的邻域坐标
        ramoff_xy = np.random.randint(-1, 2, size=(2, self.defaultNbSamples, height, width))
        # ramoff_x=np.random.randint(-1,2,size=(self.defaultNbSamples,2,height,width))

        # xr_=np.zeros((height,width))
        xr_ = np.tile(np.arange(width), (height, 1))
        # yr_=np.zeros((height,width))
        yr_ = np.tile(np.arange(height), (width, 1)).T

        xyr_ = np.zeros((2, self.defaultNbSamples, height, width))
        for i in range(self.defaultNbSamples):
            xyr_[1, i] = xr_
            xyr_[0, i] = yr_

        xyr_ = xyr_ + ramoff_xy

        xyr_[xyr_ < 0] = 0
        tpr_ = xyr_[1, :, :, -1]
        tpr_[tpr_ >= width] = width - 1
        tpb_ = xyr_[0, :, -1, :]
        tpb_[tpb_ >= height] = height - 1
        xyr_[0, :, -1, :] = tpb_
        xyr_[1, :, :, -1] = tpr_

        # xyr=np.transpose(xyr_,(2,3,1,0))
        xyr = xyr_.astype(int)
        self.samples = img[xyr[0, :, :, :], xyr[1, :, :, :]]

    def ProcessFirstFrame(self, img):
        '''
        处理视频的第一帧
        1、初始化每个像素的样本集矩阵
        2、初始化前景矩阵的mask
        3、初始化前景像素的检测次数矩阵
        参数：
        img: 传入的numpy图像素组，要求灰度图像
        返回值：
        每个像素的样本集numpy数组
        '''
        self.__buildNeighborArray(img)
        self.fgCount = np.zeros(img.shape)  # 每个像素被检测为前景的次数
        self.fgMask = np.zeros(img.shape)  # 保存前景像素

    def Update(self, img):
        '''
        处理每帧视频，更新运动前景，并更新样本集。该函数是本类的主函数
        输入：灰度图像
        '''
        height, width = img.shape
        # 计算当前像素值与样本库中值之差小于阀值范围RADIUS的个数，采用numpy的广播方法
        dist = np.abs((self.samples.astype(float) - img.astype(float)).astype(int))
        dist[dist < self.defaultRadius] = 1
        dist[dist >= self.defaultRadius] = 0
        matches = np.sum(dist, axis=0)
        # 如果大于匹配数量阀值，则是背景，matches值False,否则为前景，值True
        matches = matches < self.defaultReqMatches
        self.fgMask[matches] = self.foreground
        self.fgMask[~matches] = self.background
        # 前景像素计数+1,背景像素的计数设置为0
        self.fgCount[matches] = self.fgCount[matches] + 1
        self.fgCount[~matches] = 0
        # 如果某个像素连续50次被检测为前景，则认为一块静止区域被误判为运动，将其更新为背景点
        fakeFG = self.fgCount > 50
        matches[fakeFG] = False
        # 此处是该更新函数的关键
        # 更新背景像素的样本集，分两个步骤
        # 1、每个背景像素有1/self.defaultSubsamplingFactor几率更新自己的样本集
        ##更新样本集方式为随机选取该像素样本集中的一个元素，更新为当前像素的值
        # 2、每个背景像素有1/self.defaultSubsamplingFactor几率更新邻域的样本集
        ##更新邻域样本集方式为随机选取一个邻域点，并在该邻域点的样本集中随机选择一个更新为当前像素值
        # 更新自己样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor, size=img.shape)  # 生成每个像素的更新几率
        upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upSelfSamplesInd = np.where(upfactor == 0)  # 满足更新自己样本集像素的索引
        upSelfSamplesPosition = np.random.randint(self.defaultNbSamples,
                                                  size=upSelfSamplesInd[0].shape)  # 生成随机更新自己样本集的的索引
        samInd = (upSelfSamplesPosition, upSelfSamplesInd[0], upSelfSamplesInd[1])
        self.samples[samInd] = img[upSelfSamplesInd]  # 更新自己样本集中的一个样本为本次图像中对应像素值

        # 更新邻域样本集
        upfactor = np.random.randint(self.defaultSubsamplingFactor, size=img.shape)  # 生成每个像素的更新几率
        upfactor[matches] = 100  # 前景像素设置为100,其实可以是任何非零值，表示前景像素不需要更新样本集
        upNbSamplesInd = np.where(upfactor == 0)  # 满足更新邻域样本集背景像素的索引
        nbnums = upNbSamplesInd[0].shape[0]
        ramNbOffset = np.random.randint(-1, 2, size=(2, nbnums))  # 分别是X和Y坐标的偏移
        nbXY = np.stack(upNbSamplesInd)
        nbXY += ramNbOffset
        nbXY[nbXY < 0] = 0
        nbXY[0, nbXY[0, :] >= height] = height - 1
        nbXY[1, nbXY[1, :] >= width] = width - 1
        nbSPos = np.random.randint(self.defaultNbSamples, size=nbnums)
        nbSamInd = (nbSPos, nbXY[0], nbXY[1])
        self.samples[nbSamInd] = img[upNbSamplesInd]

    def getFGMask(self):
        '''
        返回前景mask
        '''
        return self.fgMask


class Movetracks:
    def __init__(self, block_nums=64, block_size=64):
        self.blocks = None
        self.block_nums = block_nums
        self.block_size = block_size
        self.prev_smoke_pixels_counts = None
        self.blocks_mark = np.zeros([8, 8])
        self.diff = np.zeros([8, 8])

    def Pixel_counts(self, segMat):
        segMat = cv2.resize(segMat, [512, 512])
        h, w = segMat.shape
        block_size = self.block_size
        self.blocks = [(i, j, segMat[i:i + block_size, j:j + block_size]) for i in range(0, h, block_size) for j in
                       range(0, w, block_size)]
        smoke_pixel_counts = np.array([np.sum(block[2] == 255) for block in self.blocks])
        if self.prev_smoke_pixels_counts is None:
            self.prev_smoke_pixels_counts = np.zeros_like(smoke_pixel_counts)
        diff = smoke_pixel_counts - self.prev_smoke_pixels_counts
        self.prev_smoke_pixels_counts = smoke_pixel_counts
        self.diff = diff.reshape(8, 8)
        direct = self.vector()
        return direct

    def vector(self):
        # 初始化最大值和最小值，以及它们的位置
        max_positive_value = float('-inf')
        max_negative_value = float('inf')
        max_positive_pos = (None, None)
        max_negative_pos = (None, None)

        # 遍历blocks_mark寻找最大正值和最大负值及其位置
        for i in range(self.diff.shape[0]):
            for j in range(self.diff.shape[1]):
                value = self.diff[i, j]
                if value > max_positive_value:
                    max_positive_value = value
                    max_positive_pos = (i, j)
                elif value < max_negative_value:
                    max_negative_value = value
                    max_negative_pos = (i, j)

        # 确保找到了有效的正负值
        if max_positive_pos[0] is not None and max_negative_pos[0] is not None:
            # 计算方向向量
            direction_vector = (max_negative_pos[0] - max_positive_pos[0], max_negative_pos[1] - max_positive_pos[1])
            return direction_vector
        else:
            # 如果没有找到有效的正负值，返回None或其他适当的值
            return None


def main():
    vc = cv2.VideoCapture("data/input/viedo/2.mp4")

    c = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vibe = ViBe()
    movetracks = Movetracks()
    vibe.ProcessFirstFrame(frame)
    # samples = np.zeros((frame.shape[0],frame.shape[1], defaultNbSamples))
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("segMat", cv2.WINDOW_NORMAL)
    with torch.no_grad():
        while rval:
            rval, frame = vc.read()
            # 将输入转为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 输出二值图
            # (segMat, samples) = update(gray, samples)
            vibe.Update(gray)
            segMat = vibe.getFGMask()
            # 　转为uint8类型
            segMat = segMat.astype(np.uint8)
            # 形态学处理模板初始化
            kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            # 开运算
            opening = cv2.morphologyEx(segMat, cv2.MORPH_OPEN, kernel1)
            # 形态学处理模板初始化
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            # 闭运算
            closed = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
            # 轮廓锚框
            contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            con_detect = None
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if 5 < w * h < 10000:  # Adjust size range as needed
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("frame", frame)
            cv2.imshow("SegMat", segMat)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 进行色彩值转换，RGB到HSV
            lower_hsv = np.array([0, 0, 160])  # 色彩范围h s v三变量的最小取值
            upper_hsv = np.array([110, 40, 255])  # 色彩范围h s v三变量的最小取值
            mask = cv2.inRange(hsv, lowerb=lower_hsv, upperb=upper_hsv)  # 进行色值去范围，取出对应的色彩范围进行过滤
            # cv2.imwrite("./result/" + str(c) + ".jpg", frame,[int(cv2.IMWRITE_PNG_STRATEGY)])
            result = cv2.bitwise_and(mask, segMat)
            print(movetracks.Pixel_counts(segMat))
            cv2.imshow("result", result)
            k = cv2.waitKey(1)
            if k == 27:
                vc.release()
                cv2.destroyAllWindows()
                break
            c = c + 1


if __name__ == '__main__':

    main()
