function sendAjaxRequest(url, type, data, successCallback, errorCallback) {
    $.ajax({
        url: url,
        type: type,
        contentType: 'application/json',
        data: JSON.stringify(data),
        success: successCallback,
        error: errorCallback
    });
}

function clearlog() {
    sendAjaxRequest('/clear_log', 'POST', {}, function(data) {
        if (data.status === 'success') {
            $('#responseMessage').text('日志已清除');
        } else {
            $('#responseMessage').text('清除日志失败');
        }
    }, function(jqXHR, textStatus, errorThrown) {
        console.error('请求失败:', textStatus, errorThrown);
        $('#responseMessage').text('发生错误');
    });
    return false;
}

function DT() {
    sendAjaxRequest('/delay_detection', 'POST', {}, function(response) {
        console.log("检测状态已切换", response);
        updateToggleButton1(response.js_detection_paused);
    }, function(jqXHR, textStatus, errorThrown) {
        console.error('请求失败:', textStatus, errorThrown);
    });
    return false;
}

function Vibe_paused() {
    sendAjaxRequest('/Vibe_paused', 'POST', {}, function(response) {
        console.log("检测状态已切换", response);
        updateToggleButton2(response.js_vibe_paused);
    }, function(jqXHR, textStatus, errorThrown) {
        console.error('请求失败:', textStatus, errorThrown);
    });
    return false;
}

function updateToggleButton1(isDetectionPaused) {
    var toggleButton = $('#delaydetect');
    if (isDetectionPaused) {
        toggleButton.val('继续yolo检测');
    } else {
        toggleButton.val('暂停yolo检测');
    }
}

function updateToggleButton2(isVibePaused) {
    var toggleButton = $('#Vibepaused');
    if (isVibePaused) {
        toggleButton.val('继续Vibe检测');
    } else {
        toggleButton.val('暂停Vibe检测');
    }
}

$(document).ready(function() {
    // 定期发起 AJAX 请求获取实时更新的数据
    setInterval(function() {
        $.ajax({
            url: '/alarm_act',
            method: 'GET',
            dataType: 'json',
            success: function(response) {
                // 在成功获取数据后更新元素内容
                if (response.js_alarm) {
                    $('#alarm-status').text('检测到烟雾');
                    $('#alarm-status').removeClass('normal').addClass('alarm');
                    // playAlarmSound();
                } else {
                    $('#alarm-status').text('正常');
                    $('#alarm-status').removeClass('alarm').addClass('normal');
                }
            },
            error: function(jqXHR, textStatus, errorThrown) {
                console.error('Error occurred while fetching data:', textStatus, errorThrown);
            }
        });
    }, 1000); // 每1秒更新一次
});
