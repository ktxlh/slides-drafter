$(function () {
    var datatable = $('#dataTable').DataTable( {
        "columnDefs": [
            { "width": "10%", "targets": 0 },
            { "width": "15%", "targets": 2 },
            { "width": "25%", "targets": 3 },
            { "width": "10%", "targets": 4 }
        ]
    } );

    get_table_content = function () {
        $.ajax({
            url: "/models",
            method: "GET",
            dataType: "json",
            async: true,
            success: function (data) {
                table_body = $("#table-body");
                table_body.html("");
                for(i in data){
                    item = data[i];
                    if(item['deployed'] == 'true'){
                        item['deployed'] = '<i class="fa fa-check-circle"></i>';
                    }else{
                        item['deployed'] = '';
                    }
                    datatable.row.add([
                        "<input type=\"radio\" name=\"dataRadio\">",
                        item['name'],
                        item['language'],
                        item['createTime'],
                        item['deployed']
                    ]);
                }
                datatable.draw();
            },
            error: function (data){
                console.log("fail");
            }
        });
    };
    get_table_content()

    $("#deploy-btn").click(function () {
        $("#deploy-info").css("visibility", "hidden");
        selected = $('input[type="radio"]:checked');
        dataName = selected.parent().next("td").html();
        language = selected.parent().next("td").next("td").html();
        if(dataName == undefined) {
            $("#deploy-info").css("visibility", "visible")
                .html("未选择模型文件");
        }else{
            deploy_data = {dataname: dataName, language: language};
            $.ajax({
                url: "/deploy-model",
                method: "POST",
                data: deploy_data,
                dataType: "text",
                async: true,
                success: function (data) {
                    window.location.href = "/deploy";
                }
            });
        }

    });

    $("#delete-btn").click(function () {
        $("#deploy-info").css("visibility", "hidden");
        selected = $('input[type="radio"]:checked');
        dataName = selected.parent().next("td").html();
        language = selected.parent().next("td").next("td").html();
        isDeployed = selected.parent().next("td").next("td").next("td").next("td").html();
        if(dataName == undefined) {
            $("#deploy-info").css("visibility", "visible")
                .html("未选择模型文件");
        }else if(isDeployed != ''){
            $("#deploy-info").css("visibility", "visible")
                .html("正在用于部署的择模型文件不可删除");
        }else{
            deploy_data = {'dataname': dataName, 'language': language};
            $.ajax({
                url: "/delete-model",
                method: "POST",
                data: deploy_data,
                dataType: "text",
                async: true,
                success: function (data) {
                    $("#infoModal .modal-body").html("model deleted");
                    $("#infoModal").modal("show");
                    datatable.rows().remove();
                    get_table_content();
                    // window.location.href = "/deploy";
                }
            });
        }
    });

    $("#undeploy-btn").click(function () {

        $("#deploy-info").css("visibility", "hidden");
        selected = $('input[type="radio"]:checked');
        dataName = selected.parent().next("td").html();
        language = selected.parent().next("td").next("td").html();
        isDeployed = selected.parent().next("td").next("td").next("td").next("td").html();
        if(dataName == undefined) {
            $("#deploy-info").css("visibility", "visible")
                .html("未选择模型文件");
        }else if(isDeployed == "false"){
            $("#deploy-info").css("visibility", "visible")
                .html("该数据未被部署");
        }else{
            deploy_data = {dataname: dataName, language: language};
            $.ajax({
                url: "/undeploy-model",
                method: "POST",
                data: deploy_data,
                dataType: "text",
                async: true,
                success: function (data) {
                    window.location.href = "/deploy";
                }
            });
        }
    });

});