$(function(){

    var ctx = document.getElementById("myAreaChart");
    var myLineChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: "final_loss",
                    lineTension: 0.3,
                    backgroundColor: "rgba(2,117,216,0.2)",
                    borderColor: "rgba(2,117,216,1)",
                    pointRadius: 3,
                    pointBackgroundColor: "rgba(2,117,216,1)",
                    pointBorderColor: "rgba(255,255,255,0.8)",
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "rgba(2,117,216,1)",
                    pointHitRadius: 20,
                    pointBorderWidth: 1,
                    data: [],
                },
                {
                    label: "attention_loss",
                    lineTension: 0.3,
                    backgroundColor: "rgba(216,117,2,0.2)",
                    borderColor: "rgba(216,117,2,1)",
                    pointRadius: 3,
                    pointBackgroundColor: "rgba(216,117,2,1)",
                    pointBorderColor: "rgba(255,255,255,0.8)",
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "rgba(216,117,2,1)",
                    pointHitRadius: 20,
                    pointBorderWidth: 1,
                    data: [],
                    // fill only between 1 and 0
                    fill: 0
                }
            ],
        },
        options: {
            scales: {
                xAxes: [{
                    time: {
                        // unit: 'date'
                    },
                    gridLines: {
                        display: true
                    },
                    ticks: {
                        maxTicksLimit: 7
                    }
                }],
                yAxes: [{
                    ticks: {
                        min: 0,
                        maxTicksLimit: 5
                    },
                    gridLines: {
                        color: "rgba(0, 0, 0, .125)",
                    }
                }],
            },
            legend: {
                display: true
            }
        }
    });
    var datatable = $('#dataTable').DataTable( {
        "columnDefs": [
            { "width": "10%", "targets": 0 },
            { "width": "15%", "targets": 2 },
            { "width": "25%", "targets": 3 }
        ]
    } );

    get_table_content = function () {
        $.ajax({
            url: '/seq-table',
            type: 'GET',
            dataType: 'json',
            async: true,
            success:function (data) {
                table_body = $("#table-body");
                table_body.html("");
                for(i in data){
                    item = data[i];
                    datatable.row.add([
                        "<input type=\"radio\" name=\"dataRadio\">",
                        item['tableName'],
                        item['language'],
                        item['createTime']
                    ]);
                }
                datatable.draw();
            },
            error: function (data){
                console.log("fail");
            }
        });
    };
    get_table_content();

    $.ajax({
        url: '/is-train',
        type: 'GET',
        dataType: 'json',
        async: true,
        success:function (data) {
            if(data['isTraining'] > 0){
                $("#progress-alert").html("<div class=\"alert alert-primary\" role=\"alert\"> A processing is working. Click to show details </div>");
                $("#progressModalLabel").html(data['trainName']);
                setTimeout(getProgress);
            }
        }
    });

    $("#progress-alert").click(function () {
        $("#progressModal").modal("show");
    });

    $("#train-btn").click(function () {
        $("#train-info").css("visibility", "hidden");
        selected = $('input[type="radio"]:checked');
        tableName = selected.parent().next("td").html();
        if(tableName == undefined){
            $("#train-info").css("visibility", "visible")
                .html("未选择数据");
        }else{
            $("#trainModal").modal("show");
            $("#batchSize-info").css("visibility", "hidden");
            $("#epoch-info").css("visibility", "hidden");
        }
    });

    $("#train-confirm-button").click(function () {
        batchSize = $("#batchSize").val();
        epoch = $("#epoch").val();
        if(batchSize != '' && (isNaN(batchSize) || batchSize < 0 || batchSize%1 != 0)){
            batchSize = '';
        }
        if(epoch != '' && (isNaN(epoch) || epoch < 0 || batchSize%1 != 0)){
            epoch = '';
        }
        console.log(batchSize + " " + epoch);
        selected = $('input[type="radio"]:checked');
        tableName = selected.parent().next("td").html();
        language = selected.parent().next("td").next("td").html();

        training_data = {'tableName': tableName, 'language': language, 'batchSize': batchSize, 'epoch': epoch};
        $("#progressModal").modal("show");
        $("#progress-alert").html("<div class=\"alert alert-primary\" role=\"alert\"> A processing is working. Click to show details </div>");
        $("#progressModalLabel").html(tableName);
        $("#batchSize").val("");
        $("#epoch").val("");
        setTimeout(getProgress, 1000);
        $.ajax({
            url: '/seq-train',
            type: 'POST',
            dataType: 'text',
            async: true,
            data: training_data,
            success:function (data) {

            },
            error: function (data){

            }
        });

    });

    $("#delete-btn").click(function () {
        $("#train-info").css("visibility", "hidden");
        selected = $('input[type="radio"]:checked');
        tableName = selected.parent().next("td").html();
        language = selected.parent().next("td").next("td").html();
        if(tableName == undefined){
            $("#train-info").css("visibility", "visible")
                .html("未选择数据");
        }else{
            training_data = {'tableName': tableName, 'language': language};
            $.ajax({
                url: '/seq-delete',
                type: 'POST',
                dataType: 'text',
                async: true,
                data: training_data,
                success:function (data) {
                    console.log(data + " deleted successfully");
                    $("#infoModal .modal-body").html(data + " deleted successfully");
                    $("#infoModal").modal("show");
                    datatable.rows().remove();
                    get_table_content();
                },
                error: function (data){
                    console.log("delete failed");
                }
            });
        }
    });

    getProgress = function(){
        $.ajax({
            url: '/train-progress',
            type: 'GET',
            dataType: 'json',
            async: true,
            success:function (data) {
                process = data['process'];
                final_loss = data['finalLoss'];
                attention_loss = data['attnLoss'];
                console.log(data);
                if(isNaN(process)){
                    console.log("fail: " + process);
                    $("#train-process").css("width", "0%").html("0%");
                    myLineChart.data.datasets[0].data = [];
                    myLineChart.data.datasets[1].data = [];
                    myLineChart.data.labels = new Array(0);
                    myLineChart.update();
                    $("#progressModalLabel").html("");
                    $("#progressModal").modal('hide');
                    $("#infoModal").modal("show");
                    $("#infoModal .modal-body").html(process);
                    $("#progress-alert").html("");
                }else if(process < 100){
                    $("#train-process").css("width", process + "%").html(process + "%");
                    myLineChart.data.datasets[0].data = final_loss;
                    myLineChart.data.datasets[1].data = attention_loss;
                    myLineChart.data.labels = new Array(final_loss.length);
                    myLineChart.update();
                    setTimeout(getProgress, 2000);

                }else{
                    $("#train-process").css("width", "0%").html("0%");
                    myLineChart.data.datasets[0].data = [];
                    myLineChart.data.datasets[1].data = [];
                    myLineChart.data.labels = new Array(0);
                    $("#progress-alert").html("");
                    $("#progressModal").modal("hide");
                    $("#infoModal").modal("show");
                    $("#infoModalLabel").html("Training Completed")
                    $("#infoModal .modal-body").html("RNN loss: " + final_loss[final_loss.length-1] + "<br>" +
                        "Attention loss: " + (attention_loss[attention_loss.length-1] - final_loss[final_loss.length-1])/2);
                }
            },
            error: function (data) {
                $("#train-process").css("width", "0%").html("0%");
                myLineChart.data.datasets[0].data = [];
                myLineChart.data.datasets[1].data = [];
                myLineChart.data.labels = new Array(0);
                $("#progressModalLabel").html("");
                $("#progressModal").modal('hide');
                $("#infoModal").modal("show");
                $("#infoModal .modal-body").html("Training Failed");
                $("#progress-alert").html("");
            }
        });
    };


});