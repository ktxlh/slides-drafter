$(function(){

    rand_text = ['Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet\'s leading subsidiary and will continue to be the umbrella company for Alphabet\'s Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet.',
        'Deep learning (also known as deep structured learning or hierarchical learning) is part of a broader family of machine learning methods based on artificial neural networks. Learning can be supervised, semi-supervised or unsupervised.\n' +
        'Deep learning architectures such as deep neural networks, deep belief networks, recurrent neural networks and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, audio recognition, social network filtering, machine translation, bioinformatics, drug design, medical image analysis, material inspection and board game programs, where they have produced results comparable to and in some cases superior to human experts.'];

    filename = '';

    $("#download_btn").attr('disabled', true);

    $('#upload_btn').click(function () {
        $("#download_btn").attr('disabled', true);
        text = $('#inputtext').val();
        if(text == '') {
            alert("Please input text");
        }else {
            $.ajax({
                url: '/analyze',
                type: 'POST',
                data: {input: text},
                success: function (data) {

                    filename=data;
                    $("#filename").val(filename);
                    $("#download_btn").attr('disabled', false);
                }
            });
        }

    });

    $('#inputtext').bind("input propertychange", function (event) {
        $("#download_btn").attr('disabled', true);
    })
    
    $('#random_btn').click(function () {
        r = Math.floor(Math.random() * 2);
        $('#inputtext').val(rand_text[r]);
        $("#download_btn").attr('disabled', true);
    })

});