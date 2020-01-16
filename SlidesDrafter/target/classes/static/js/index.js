$(function(){

    filename = '';

    $('#upload_btn').click(function () {
        text = $('#inputtext').val();
        $.ajax({
            url: '/analyze',
            type: 'POST',
            data: {input: text},
            success: function (data) {

                filename=data;
                $("#filename").val(filename);
                alert(data);
            }
        });
    });


});