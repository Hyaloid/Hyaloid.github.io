---
layout: post
title: 非 root 用户安装 boost
subtitle: Ubuntu-18.04 boost 安装
tags: [boost, Linux, C/C++]
categories: [Env]
comments: true
author: SeaMount
---

去 [boost 官网](https://boostorg.jfrog.io/artifactory/main/release/)选择合适的版本下载，本文安装的是 boost-1.81.0。

1. 下载 boost 源码

    ```shell
    wget https://boostorg.jfrog.io/artifactory/main/release/1.81.0/source/boost_1_81_0.tar.gz
    ```

2. 解压

    ```
    tar -xvf boost_1_81_0.tar.gz
    ```

3. 运行 `bootstrap.sh` 检查安装环境

    ```shell
    cd boost_1_81_0/
    ./bootstrap.sh
    ```

4. 构建并安装 boost 库，上一步成功之后会生成 `b2`，使用 `b2` 来构建 boost 库。

    ```
    mkdir /home/username/boost
 
    # --prefix 指定安装路径
    ./b2 --prefix=/home/username/boost install
    ```

5. 配置环境变量（可选）

    ```shell
    export BOOST_INCLUDE=/home/username/boost/include
    export BOOST_LIB=/home/username/boost/lib
    ```

6. 测试 boost 库安装是否成功

    将以下代码写入文件 `test.cpp`
    ```c
    #include <boost/date_time/gregorian/gregorian.hpp>
    #include <iostream>
    int main() 
    { 
        boost::gregorian::date d(boost::gregorian::day_clock::local_day());
        std::cout << d.year() << d.month() <<d.day() << std::endl; 
    }
    ```

    ```shell
    g++ -I /home/username/boost/include -L /home/username/boost/lib test.cpp -o test
    
    ./test 
    
    2023Dec25
    ```
