---
layout: post
title: 非 root 用户安装 gcc
subtitle: Ubuntu-18.04 gcc 安装
tags: [gcc, Linux]
categories: [Env]
comments: true
author: SeaMount
---

首先[下载](https://ftp.gnu.org/gnu/gcc/)对应的 gcc 版本，本文安装的是 gcc-11.1.0。


1. 下载源代码

    ```shell
    wget https://ftp.gnu.org/gnu/gcc/gcc-11.1.0/gcc-11.1.0.tar.gz
    ```

2. 解压文件

    ```shell
    tar -xvf gcc-11.1.0.tar.gz 
    ```

3. 下载依赖项

    ```shell
    cd gcc-11.1.0/

    # gmp,mpfr,mpc,isl 等依赖下载
    ./contrib/download_prerequisites
    ```

4. 创建 build 目录用于编译

    ```shell
    mkdir build && cd build
    ```

5. 设置编译选项，生成 make 文件

    ```shell
    ../configure --prefix=/home/username/gcc11.1 --enable-checking=release --enable-languages=c,c++ --enable-threads=posix --disable-multilib
    ```

    {: .box-note}
    --enable-checking=release，增加 release 的检查，也可以使用 `--disable-checking` 使编译过程中不做额外检查 \
    --enable-languages 表示你要让你的 gcc 支持哪些语言 \
    --disable-multilib 不生成编译为其他平台可执行代码的交叉编译器

6. 编译

    ```shell
    make -j48

    # 编译检查，确保编译无误
    make check
    ```

7. 安装

    ```shell
    make install
    ```

8. 编译项目代码时加上 CMake 编译的 gcc 路径即可

    ```shell
    cmake -DCMAKE_C_COMPILER=/home/username/gcc11.1.0/bin/gcc -DCMAKE_CXX_COMPILER=/home/username/gcc11.1.0/bin/g++ ..
    ```