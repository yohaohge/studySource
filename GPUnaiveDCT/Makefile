CC = nvcc
CFLAGS = -O3 
LDFLAGS = -lm

all: mjpeg_gpu

mjpeg_gpu: mjpeg_encoder.o dct_quantize.o
	$(CC) $(CFLAGS) dct_quantize.o mjpeg_encoder.o -o mjpeg_gpu

dct_quantize.o: dct_quantize.cu
	$(CC) $(CFLAGS) dct_quantize.cu -c $(LDFLAGS)

clean:
	rm -f *.o mjpeg_gpu
