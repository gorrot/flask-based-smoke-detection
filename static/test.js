
function sayHello() {
     var button = document.getElementById('myButton');
    button.addEventListener('click', function(event) {
        event.preventDefault(); // 阻止按钮点击事件的默认行为
        alert('警报消息'); // 触发警报消息
    });
 }