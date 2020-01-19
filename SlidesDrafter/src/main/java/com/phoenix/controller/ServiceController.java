package com.phoenix.controller;

import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.ResponseBody;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;

@Controller
@EnableAutoConfiguration
public class ServiceController {
    private static String base_dir;
    private static String python_dir;
    private static String anaconda_pyhont_dir;
    private static String input_dir;
    private static String seg_dir;
    private static String slides_dir;
    private static String model_dir;
    private static String filename;
    static {
        base_dir = new File(System.getProperty("user.dir")).getAbsolutePath();
        input_dir = base_dir + "/input_text/";
        seg_dir = base_dir + "/seg_text/";
        slides_dir = base_dir + "/generated_slides/";
        model_dir = base_dir + "/model/";
        python_dir = "python";
        anaconda_pyhont_dir = "python";
        filename = "";
    }


    @RequestMapping(value = "/analyze", method = RequestMethod.POST)
    @ResponseBody
    String analyze(String input) {

        if(input.length() == 0) {
            filename = "empty";
        } else {
            filename = input.split(" ")[0];
        }

        //Save as file
        try{
            PrintStream ps = new PrintStream(new FileOutputStream(input_dir + filename));
            ps.print(input);
            ps.close();
        }catch (Exception e){
            e.printStackTrace();
            System.out.println("save file failed");
            return "{}";
        }
        //Seg
        System.out.println("Text to Segment");
        String[] seg_arguments = new String[]{
                python_dir,
                model_dir + "/text-seg/seg.py",
                "-input_file",
                input_dir + filename,
                "-output_file",
                seg_dir + filename,
                "-base_dir",
                model_dir
        };

        try{
            Process process = Runtime.getRuntime().exec(seg_arguments);
            process.waitFor();
        }catch(Exception e){
            e.printStackTrace();
        }
        for(String str: seg_arguments) {
            System.out.print(str + " ");
        }
        System.out.println();

        //slides
        System.out.println("Slides Gen");
        String[] slides_arguments = new String[]{
                python_dir,
                model_dir + "/slides/slidespy1.py",
                "-input_file",
                seg_dir + filename,
                "-output_file",
                slides_dir + filename + ".pptx"
        };

        for(String str: slides_arguments) {
            System.out.print(str + " ");
        }
        System.out.println();

        try{
            Process process = Runtime.getRuntime().exec(slides_arguments);
            process.waitFor();
        }catch(Exception e){
            e.printStackTrace();
        }

        System.out.println(filename + " finished!");
        return filename;
    }


    @RequestMapping("downloadFileAction")
    public void downloadFileAction(HttpServletRequest request, HttpServletResponse response) {

        response.setCharacterEncoding(request.getCharacterEncoding());
        response.setContentType("application/octet-stream");
        FileInputStream fis = null;
        try {
            File file = new File(slides_dir + filename + ".pptx");
            fis = new FileInputStream(file);
            response.setHeader("Content-Disposition", "attachment; filename="+file.getName());
            IOUtils.copy(fis,response.getOutputStream());
            response.flushBuffer();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fis != null) {
                try {
                    fis.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
