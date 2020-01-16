package com.phoenix;

import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.web.servlet.MultipartConfigFactory;
import org.springframework.context.ConfigurableApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;

import javax.servlet.MultipartConfigElement;
import javax.xml.ws.Service;


import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.Statement;

import static org.springframework.boot.SpringApplication.run;

@SpringBootApplication
@Configuration
public class Application {
    public static void main(String[] args) {
        ConfigurableApplicationContext run = run(Application.class, args);

    }

    @Bean
    public MultipartConfigElement multipartConfigElement() {
        MultipartConfigFactory factory = new MultipartConfigFactory();
        //单个文件最大
        factory.setMaxFileSize("10000MB"); //KB,MB
        /// 设置总上传数据总大小
        factory.setMaxRequestSize("10000MB");
        return factory.createMultipartConfig();
    }
}
