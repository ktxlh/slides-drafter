$(function(){
    var codeToSend = "";
    var apiToSend = "";
    var isCodeReady = false;
    var isApiReady = false;
    Dropzone.autoDiscover = false;

// Get the template HTML and remove it from the doument
    var previewNode = document.querySelector("#template");
    previewNode.id = "";
    var previewTemplate = previewNode.parentNode.innerHTML;
//开始先删除单个文件的布局
    previewNode.parentNode.removeChild(previewNode);


// code upload
    var codeDropzone = new Dropzone(document.querySelector("#codeupload"), { // 指定拖拽区为body
        url: "/zip-upload", // Set the url
        // thumbnailWidth: 80,
        // thumbnailHeight: 80,
        previewTemplate: previewTemplate,//每个文件的html模板
        autoQueue: false, // 当队列有文件，是否立刻自动上传到服务器
        previewsContainer: "#code-previews", // 每个文件的模板放在这个区域内
        clickable: "#codeupload", // 点击某个按钮或区域后出现选择电脑中本地图片
        maxFilesize: 100000,
        timeout: 7200000,
        maxFiles: 1,
        createImageThumbnails: false,
        acceptedFiles: ".zip"
    });

    codeDropzone.on("addedfile", function(file) {
        // 让模版中的单个文件可以点击上传
        file.previewElement.querySelector(".start").onclick = function() { codeDropzone.enqueueFile(file); };
    });

    codeDropzone.on("sending", function(file) {
        // 显示整体的上传的进度条，说明：原来是0，所以上面的style.width = progress + "%"即使是100%也看不到
        // document.querySelector("#total-progress").style.opacity = "1";
        // 失效上传按钮
        file.previewElement.querySelector(".start").setAttribute("disabled", "disabled");
    });

    codeDropzone.on("success", function (file) {
        codeToSend = file.name;
        isCodeReady = true;
    });

    codeDropzone.on("removedfile", function (file) {
        codeToSend = "";
        isCodeReady = false;
    });

// api upload
    var apiDropzone = new Dropzone(document.querySelector("#apiupload"), { // 指定拖拽区为body
        url: "/api-upload", // Set the url
        // thumbnailWidth: 80,
        // thumbnailHeight: 80,
        previewTemplate: previewTemplate,//每个文件的html模板
        autoQueue: false, // 当队列有文件，是否立刻自动上传到服务器
        previewsContainer: "#api-previews", // 每个文件的模板放在这个区域内
        clickable: "#apiupload", // 点击某个按钮或区域后出现选择电脑中本地图片
        maxFilesize: 100000,
        timeout: 7200000,
        maxFiles: 1,
        createImageThumbnails: false
    });

    apiDropzone.on("addedfile", function(file) {
        // 让模版中的单个文件可以点击上传
        file.previewElement.querySelector(".start").onclick = function() { apiDropzone.enqueueFile(file); };
    });

    apiDropzone.on("sending", function(file) {
        // 显示整体的上传的进度条，说明：原来是0，所以上面的style.width = progress + "%"即使是100%也看不到
        // document.querySelector("#total-progress").style.opacity = "1";
        // 失效上传按钮
        file.previewElement.querySelector(".start").setAttribute("disabled", "disabled");
    });

    apiDropzone.on("success", function (file) {
        apiToSend = file.name;
        isApiReady = true;
    });

    apiDropzone.on("removedfile", function (file) {
        apiToSend = "";
        isApiReady = false;
    });

    $.ajax({
        url: '/is-create',
        type: 'GET',
        dataType: 'json',
        async: true,
        success:function (data) {
            if(data['isCreating'] > 0){
                $("#progress-alert").html("<div class=\"alert alert-primary\" role=\"alert\"> A processing is working. Click to show details </div>");
                $("#progressModalLabel").html(data['createName']);
                setTimeout(getProgress);
            }
        }
    });

    $("#progress-alert").click(function () {
        $("#progressModal").modal("show");
    });

    $("#create-btn").click(function () {
        selected = $("#languageSelect").val();
        newName = $("#newDataName").val();
        $("#file-info").css("visibility", "hidden");
        if(isCodeReady == false){
            $("#file-info").css("visibility", "visible")
                .html("请上传源代码文件");
        }else if(isApiReady == false){
            $("#file-info").css("visibility", "visible")
                .html("请上传api定义文件");
        }else if(newName.length == 0){
            $("#file-info").css("visibility", "visible")
                .html("请输入新数据集的名称");
        }else{
            $("#createModal .modal-body").html("Are you sure to create the training data of <strong>" + selected +
                "</strong> from <strong>" + codeToSend + "</strong> and <strong>" + apiToSend + "</strong>?");
            $("#createModal").modal("show");
        }
    });

    $("#create-confirm-button").click(function () {
        selected = $("#languageSelect").val();
        newName = $("#newDataName").val();
        filename = {codefile: codeToSend, apifile: apiToSend, fileType: selected, newDataName: newName};
        $("#progress-alert").html("<div class=\"alert alert-primary\" role=\"alert\"> A processing is working. Click to show details </div>");
        $("#progressModal").modal("show");
        $("#progressModalLabel").html(newName);
        setTimeout(getProgress, 1000);
        $.ajax({
            url: '/seq-create',
            type: 'POST',
            dataType: 'text',
            async: true,
            data: filename,
            success:function (data) {

            }
        });
    });

    getProgress = function(){
        $.ajax({
            url: '/create-progress',
            type: 'GET',
            dataType: 'text',
            async: true,
            success:function (data) {
                if(isNaN(data)){
                    console.log("fail: " + data);
                    // $('body').removeClass("modal-open");
                    // $("#progressModal").removeClass("show")
                    //     .css("display", "none")
                    //     .attr("aria-hidden", "true");
                    $("#progressModal").modal('hide');
                    $("#infoModal").modal("show");
                    $("#infoModal .modal-body").html(data);
                    $("#progress-alert").html("");
                    $("#create-process").css("width", "0%").html("0%");
                    $("#progressModalDetail").html("");
                }else if(data < 100){
                    console.log(data);
                    $("#create-process").css("width", data + "%").html(data + "%");
                    if(data < 5){
                        $("#progressModalDetail").html("Unzipping files");
                    }else if(data < 90){
                        $("#progressModalDetail").html("Preprocessing codes");
                    }else if(data < 97){
                        $("#progressModalDetail").html("Generating sequence tokens");
                    }else if(data < 100){
                        $("#progressModalDetail").html("Storing");
                    }
                    setTimeout(getProgress, 2000);
                }else{
                    $("#create-process").css("width", "0%").html("0%");
                    $("#progressModalDetail").html("");
                    $.ajax({
                        url: '/is-create',
                        type: 'GET',
                        dataType: 'json',
                        async: true,
                        success:function (retData) {
                            $("#progress-alert").html("");
                            $("#progressModal").modal("hide");
                            $("#infoModal").modal("show");
                            $("#infoModalLabel").html("Data Generated");
                            $("#infoModal .modal-body").html("File: " + retData['createName'] + "<br>" +
                                                            "Code files: " + retData['numCode'] + "<br>" +
                                                            "Parse successfully: " + retData['numJson'] + "<br>" +
                                                            "Training data generated: " + retData['numSeq'] + "<br>");
                        }
                    });
                }
            },
            error: function () {
                $("#progressModal").modal('hide');
                $("#infoModal").modal("show");
                $("#infoModal .modal-body").html("Preprocessing Failed");
                $("#progress-alert").html("");
            }
        });
    };
});
