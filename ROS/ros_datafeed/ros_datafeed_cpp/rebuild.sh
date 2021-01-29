#!/bin/sh
make clean
rm CMakeCache.txt
cmake .
make
