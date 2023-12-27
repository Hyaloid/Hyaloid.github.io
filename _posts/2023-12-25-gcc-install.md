---
layout: post
title: 非 root 用户安装 gcc
subtitle: Ubuntu-18.04 gcc 安装
tags: [gcc, Linux, C/C++]
categories: [Env]
comments: true
author: SeaMount
---

最近在将一个 CPP 项目转成 CUDA 项目，但是编译的时候发现现有的 CUDA 版本和 GCC 版本不匹配，需要自己安装对应版本的 GCC，但是自己又没有服务器的超级用户权限，用不了 `update-alternatives`，所以需要使用源码编译安装与 CUDA 版本相匹配的相应版本的 GCC。

<table align="center">
  <thead>
    <tr>
      <th>CUDA version</th>
  	  <th>max supported GCC version</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>11.4.1+,11.5,11.6,11.7</td>
      <td>11</td>
    </tr>
  	<tr>
  	  <td>11.1,11.2,11.3,11.4.0</td>
      <td>10</td>
  	</tr>
    <tr>
  	  <td>11</td>
      <td>9</td>
  	</tr>
  	<tr>
  	  <td>10.1,10.2 </td>
      <td>8</td>
  	</tr>
  	<tr>
  	  <td>9.2,10.0 </td>
      <td>7</td>
  	</tr>
  	<tr>
  	  <td>9.0,9.1 </td>
      <td>6</td>
  	</tr>
  	<tr>
  	  <td>8 </td>
      <td>5.3</td>
  	</tr>
  </tbody>
</table>

首先[下载](https://ftp.gnu.org/gnu/gcc/)对应的 gcc 版本，我使用的 CUDA 版本是 11.8，所以安装的 GCC 版本是 gcc-11.1.0。


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
    `--enable-checking=release`，增加 release 的检查，也可以使用 `--disable-checking` 使编译过程中不做额外检查 \
    `--enable-languages` 表示你要让你的 gcc 支持哪些语言 \
    `--disable-multilib` 不生成编译为其他平台可执行代码的交叉编译器

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