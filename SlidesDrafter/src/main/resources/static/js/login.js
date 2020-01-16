$(function(){
    $("#buttom_log").click(function () {
        var username = $("#username").val();
        var password = $("#password").val();
        $("#username-info").css("visibility", "hidden");
        $("#password-info").css("visibility", "hidden");
        if(username.length == 0){
            $("#username-info").css("visibility", "visible")
                .html("用户名不能为空");
        }else if(password.length == 0){
            $("#password-info").css("visibility", "visible")
                .html("密码不能为空");
        }else{
            var user = {username: username, password: hex_md5(password)};
            $.ajax({
                url: '/login',
                type: 'POST',
                dataType: 'text',
                async: false,
                data: user,
                success:function (data) {
                    if(data == 'success'){
                        window.location.href = '/create';
                    }else{
                        $("#password-info").css("visibility", "visible")
                            .html("密码或帐号错误");
                    }
                }
            });
        }
    });

    $("#buttom_reg").click(function () {
        window.location.href = '/register';
    })
});