#/bin/bash
make clean
make
./c63enc -w 352 -h 288 -o test.c63 foreman.yuv
gcc -S dsp.c

gprof c63enc gmon.out -p
