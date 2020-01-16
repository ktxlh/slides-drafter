$(function () {

    $("#register").click(function () {
        var username = $("#username").val();
        var password = $("#password").val();
        var passwordAgain = $("#passwordAgain").val();

        $("#username-info").css("visibility", "hidden");
        $("#password-info").css("visibility", "hidden");
        $("#passwordAgain-info").css("visibility", "hidden");

        if(username.length == 0){
            $("#username-info").css("visibility", "visible")
                            .html("用户名不能为空");

        }else if(password.length == 0) {
            $("#password-info").css("visibility", "visible")
                .html("密码不能为空");
        }
        else if(password != passwordAgain){
            $("#passwordAgain-info").css("visibility", "visible")
                .html("密码不一致");
        }else{
            user = JSON.stringify({username: username, password: hex_md5(password)});
            $.ajax({
                url: '/users',
                type: 'POST',
                dataType: 'text',
                async: false,
                data: user,
                contentType:'application/json;charset=UTF-8',
                success:function (data) {
                    if(data == 'success')
                        window.location.href= "/login";
                    else
                        $("#username-info").css("visibility", "visible")
                            .html("该用户已存在");
                }
            });
        }
    });
});