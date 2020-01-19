package com.phoenix.controller;

import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.*;

import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.ModelAndView;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.*;

@Controller
@EnableAutoConfiguration
public class MainController {

    private static String filename = "";

    @RequestMapping("/")
    public ModelAndView rootPage(){
        return new ModelAndView("index1");
    }

    @RequestMapping(value = "/index", method = RequestMethod.GET)
    public ModelAndView login(){
        return new ModelAndView("index1");
    }



}
