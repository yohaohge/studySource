#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>

#include "mjpeg_encoder.h"
#include "dct_quantize.h"

static char output_file[MAX_FILELENGTH] = DEFAULT_OUTPUT_FILE;
static float cosuv[8][8][8][8];
static FILE *outfile;

static uint32_t bit_buffer = 0;
static uint32_t bit_buffer_width = 0;

/* Moved to header, shared with GPU */
extern uint32_t yph;
extern uint32_t ypw;
extern uint32_t uph;
extern uint32_t upw;
extern uint32_t vph;
extern uint32_t vpw;

/* getopt */
extern int optind;
extern char *optarg;
static struct option long_options[] = {
                   {"help", 0, 0, '0'},
                   {"output", 1, 0, '0'},
                   {"height", 1, 0, '0'},
                   {"width", 1, 0, '0'},
                   {0, 0, 0, 0}};

/* Read YUV frames */
static yuv_t* read_yuv(const char *filename)
{
  FILE *file;
  long size = 0;
  size_t read = 0;
  yuv_t *image = malloc(sizeof(yuv_t));

  file = fopen(filename, "rb");

  if(file == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  if(fseek(file, 0, SEEK_END))
  {
    perror("fseek");
    exit(EXIT_FAILURE);
  }

  size = ftell(file);
  if(size == -1)
  {
    perror("ftell");
    exit(EXIT_FAILURE);
  }

  rewind(file);

  if(size != (int)(y_comp_size*1.5))
  {
    fprintf(stderr, "Dimensions do not match filesize! Got %ld, expectd %ld\n", size, (unsigned long)(width*height*1.5)); // XXX: *sizeof(T) ?
    exit(EXIT_FAILURE);
  }

  /* Read Y' */
  image->Y = malloc(y_comp_size);
  read += fread(image->Y, 1, y_comp_size, file);
  if(ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  /* Read U */
  image->U = malloc(width*height);
  read += fread(image->U, 1, uv_comp_size, file);
  if(ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  /* Read V */
  image->V = malloc(width*height);
  read += fread(image->V, 1, uv_comp_size, file);
  if(ferror(file))
  {
    perror("ferror");
    exit(EXIT_FAILURE);
  }

  if(read != size)
  {
    fprintf(stderr, "Read %u of %li bytes. An error has occured...\n", (uint32_t) read, size);
    exit(EXIT_FAILURE);
  }

  fclose(file);
  return image;
}

static void put_byte(int byte)
{
  int status = fputc(byte, outfile);

  if (status == EOF) { /* XXX: if(byte != status) maybe better? */
    fprintf(stderr, "Error writing byte\n");
    exit(EXIT_FAILURE);
  }
}

static void put_bytes(const void* data, unsigned int len)
{
  int n = fwrite(data, 1, len, outfile);

  if(n != len) {
    fprintf(stderr, "Error writing bytes\n");
    exit(-1);
  }
}

static void write_SOI()
{
  put_byte(0xff);
  put_byte(0xd8);
}

static void write_DQT()
{
  int16_t size = 2 + (3 * 65);

  put_byte(0xff);
  put_byte(0xdb);

  put_byte(size >> 8);
  put_byte(size & 0xff);

  put_byte(0);
  put_bytes(yquanttbl, 64);
  put_byte(1);
  put_bytes(uquanttbl, 64);
  put_byte(2);
  put_bytes(vquanttbl, 64);
}

static void write_SOF0()
{
  int16_t size = 8 + 3 * COLOR_COMPONENTS;

  /* Header marker */
  put_byte(0xff);
  put_byte(0xc0);

  /* Size of header */
  put_byte(size >> 8);
  put_byte(size & 0xff);

  /* Precision */
  put_byte(8);

  /* Width and height */
  put_byte(height >> 8);
  put_byte(height & 0xff);
  put_byte(width >> 8);
  put_byte(width & 0xff);

  put_byte(COLOR_COMPONENTS);

  put_byte(1); /* Component id */
  put_byte(0x22); /* hor | ver sampling factor */
  put_byte(0); /* Quant. tbl. id */

  put_byte(2); /* Component id */
  put_byte(0x11); /* hor | ver sampling factor */
  put_byte(1); /* Quant. tbl. id */

  put_byte(3); /* Component id */
  put_byte(0x11); /* hor | ver sampling factor */
  put_byte(2); /* Quant. tbl. id */
}

static void write_DHT_HTS(uint8_t id, uint8_t *numlength, uint8_t* data)
{
    /* Find out how many codes we are to write */
    int i;
    int n = 0;
    for(i = 0; i < 16; ++i)
        n += numlength[i];

    put_byte(id);
    put_bytes(numlength, 16);
    put_bytes(data, n);
}

static void write_DHT()
{
  int16_t size = 0x01A2; /* 2 + n*(17+mi); */

  /* Define Huffman Table marker */
  put_byte(0xff);
  put_byte(0xc4);

  /* Length of segment */
  put_byte(size >> 8);
  put_byte(size & 0xff);

  /* Write the four huffman table specifications */
  write_DHT_HTS(0x00, DCVLC_num_by_length[0], DCVLC_data[0]); /* DC table 0 */
  write_DHT_HTS(0x01, DCVLC_num_by_length[1], DCVLC_data[1]); /* DC table 1 */
  write_DHT_HTS(0x10, ACVLC_num_by_length[0], ACVLC_data[0]); /* AC table 0 */
  write_DHT_HTS(0x11, ACVLC_num_by_length[1], ACVLC_data[1]); /* AC table 1 */
}

static void write_SOS()
{
  int16_t size = 6 + 2 * COLOR_COMPONENTS;

  put_byte(0xff);
  put_byte(0xda);

  put_byte(size >> 8);
  put_byte(size & 0xff);

  put_byte(COLOR_COMPONENTS);

  put_byte(1); /* Component id */
  put_byte(0x00); /* DC | AC huff tbl */
  put_byte(2); /* Component id */
  put_byte(0x11); /* DC | AC huff tbl */
  put_byte(3); /* Component id */
  put_byte(0x11); /* DC | AC huff tbl */
  put_byte(0); /* ss, first AC */
  put_byte(63); /* se, last AC */
  put_byte(0); /* ah | al */
}

static void write_EOI()
{
  put_byte(0xff);
  put_byte(0xd9);
}

static inline uint8_t bit_width(int16_t i)
{
  return (uint8_t) (ceil((log(abs(i)+1))*M_LOG2E));
}

/**
 * Adds a bit to the bitBuffer. A call to Flush() is needed
 * in order to write any remainding bits in the buffer before
 * writing using another function.
 */
static void put_bits(int16_t bits, uint8_t n)
{
  assert(n <= 24  && "Error writing bit");

  if(n == 0)
    return;

  bit_buffer <<= n;
  bit_buffer |= bits & ((1 << n) - 1);
  bit_buffer_width += n;

  while(bit_buffer_width >= 8) {
    uint8_t c = (uint8_t)(bit_buffer >> (bit_buffer_width - 8));  //FIXME: int8_t or int16_t here?
    put_byte(c);
    if(c == 0xff)
      put_byte(0);
    bit_buffer_width -= 8;
  }
}


/**
 * Flushes the bitBuffer by writing zeroes to fill a full byte
 */
static void flush()
{
  if(bit_buffer > 0) {
    uint8_t c = bit_buffer << (8 - bit_buffer_width);
    put_byte(c);
    if(c == 0xff)
      put_byte(0);
  }

  bit_buffer = 0;
  bit_buffer_width = 0;
}

static void write_block(int16_t *in_data, uint32_t width, uint32_t height,
			uint32_t uoffset, uint32_t voffset, int16_t *prev_DC,
			int32_t cc)
{
  uint32_t zigzag, i, j;

  static int16_t block[64];
  int32_t num_ac = 0;

  /* ZigZag */
  for(zigzag = 0; zigzag < 64; zigzag++)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];
    block[zigzag] = in_data[(voffset+v)*width+(uoffset+u)];
  }

  /* Calculate DC component, and write to stream */
  int16_t dc = block[0] - *prev_DC;
  *prev_DC = block[0];
  uint8_t size = bit_width(dc);
  put_bits(DCVLC[cc][size],DCVLC_Size[cc][size]);
  if(dc < 0)
    dc = dc - 1;

  put_bits(dc, size);

  /* find the last nonzero entry of the ac-coefficients */
  for(j = 64; j > 1 && !block[j-1]; j--)
    ;

  /* Put the nonzero ac-coefficients */
  for(i = 1; i < j; i++)
  {
    int16_t ac = block[i];
    if(ac == 0)
    {
      if(++num_ac == 16)
      {
	put_bits(ACVLC[cc][15][0], ACVLC_Size[cc][15][0]);
	num_ac = 0;
      }
    }
    else
    {
      uint8_t size = bit_width(ac);
      put_bits(ACVLC[cc][num_ac][size], ACVLC_Size[cc][num_ac][size]);

      if(ac < 0)
	--ac;

      put_bits(ac, size);

      num_ac = 0;
    }
  }

  /* Put end of block marker */
  if(j < 64)
    put_bits(ACVLC[cc][0][0], ACVLC_Size[cc][0][0]);
}

static void write_interleaved_data_MCU(int16_t *dct, uint32_t wi, uint32_t he,
				       uint32_t h, uint32_t v, uint32_t x,
				       uint32_t y, int16_t *prev_DC, int32_t cc)
{
  uint32_t i, j, ii, jj;
  for(j = y*v*8; j < (y+1)*v*8; j += 8)
  {
    jj = he-8;
    jj = MIN(j, jj);

    for(i = x*h*8; i < (x+1)*h*8; i += 8)
    {
       ii = wi-8;
       ii = MIN(i, ii);

      write_block(dct, wi, he, ii, jj, prev_DC, cc);
    }
  }
}

static void write_interleaved_data(dct_t *out)
{
  int16_t prev_DC[3] = {0, 0, 0};
  uint32_t u, v;

  /* Set up which huffman tables we want to use */
  int32_t yhtbl = 0;
  int32_t uhtbl = 1;
  int32_t vhtbl = 1;

  /* Find the number of MCU's for the intensity */
  uint32_t ublocks = (uint32_t) (ceil(ypw/(float)(8.0f*YX)));
  uint32_t vblocks = (uint32_t) (ceil(yph/(float)(8.0f*YY)));

  /* Write the MCU's interleaved */
  for(v = 0; v < vblocks; ++v)
  {
    for(u = 0; u < ublocks; ++u)
    {
      write_interleaved_data_MCU(out->Ydct, ypw, yph, YX, YY, u, v, &prev_DC[0], yhtbl);
      write_interleaved_data_MCU(out->Udct, upw, uph, UX, UY, u, v, &prev_DC[1], uhtbl);
      write_interleaved_data_MCU(out->Vdct, vpw, vph, VX, VY, u, v, &prev_DC[2], vhtbl);
    }
  }

  flush();
}


static void encode(yuv_t *image, int pic_nr)
{
  dct_t *out = malloc(sizeof(dct_t));
  out->Ydct = malloc(yph*ypw*(sizeof(*out->Ydct)));
  out->Udct = malloc(uph*upw*(sizeof(*out->Udct)));
  out->Vdct = malloc(vph*vpw*(sizeof(*out->Vdct)));

  memset(out->Ydct, 0, sizeof(int16_t) * yph * ypw);
  memset(out->Udct, 0, sizeof(int16_t) * uph * upw);
  memset(out->Vdct, 0, sizeof(int16_t) * vph * vpw);

  /* We have this nice GPU up and running! Let's do some DCT! */
  gpu_dct_quantize(image, out);

  /* Write headers */
  /* Start Of Image */
  write_SOI();
  /* Define Quantization Table(s) */
  write_DQT();
  /* Start Of Frame 0(Baseline DCT) */
  write_SOF0();
  /* Define Huffman Tables(s) */
  write_DHT();
  /* Start of Scan */
  write_SOS();

  write_interleaved_data(out);

  /* End Of Image */
  write_EOI();

  free(out->Ydct);
  free(out->Udct);
  free(out->Vdct);
  free(out);
}

static void print_help()
{
  fprintf(stderr, "Usage: ./gpu_mjpeg [options] input_files\n");
  fprintf(stderr, "Commandline options:\n");
  fprintf(stderr, "  --help                         produce help message\n");
  fprintf(stderr, "  -h [ --height ]                Required: height of images to compress\n");
  fprintf(stderr, "  -w [ --width ]                  Required: width of images to compress\n");
  fprintf(stderr, "  -o [ --outfile ] arg (=a.mjpg) filename to compress to\n");
  fprintf(stderr, "\n");
  exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
  int c;
  int option_index = 0;
  int tmp;
  yuv_t *image;

  if(argc == 1)
  {
    print_help();
    exit(EXIT_FAILURE);
  }

  while((c = getopt_long(argc, argv, "h:w:o", long_options, &option_index)) != -1)
  {
    switch(c)
    {
      case 'h':
		height = atoi(optarg);
		break;
      case 'w':
		width = atoi(optarg);
		break;
      case 'o':
		strncpy(output_file,argv[optind++],MAX_FILELENGTH);
		break;
      default:
		print_help();
		break;
    }
  }

  if(optind >= argc)
  {
    fprintf(stderr, "Error getting program options, try --help.\n");
    exit(EXIT_FAILURE);
  }

  outfile = fopen(output_file, "wb");
  if(outfile == NULL)
  {
    perror("fopen");
    exit(EXIT_FAILURE);
  }

  /* Calculate the padded width and height */
  ypw = (uint32_t)(ceil(width/8.0f)*8);
  yph = (uint32_t)(ceil(height/8.0f)*8);
  upw = (uint32_t)(ceil(width*UX/(YX*8.0f))*8);
  uph = (uint32_t)(ceil(height*UY/(YY*8.0f))*8);
  vpw = (uint32_t)(ceil(width*VX/(YX*8.0f))*8);
  vph = (uint32_t)(ceil(height*VY/(YY*8.0f))*8);

  /* Divide u and v by 4 */
  y_comp_size	  = width * height;
  uv_comp_size	  = y_comp_size >> 2;

  /* INIT GPU - Gentlemen, please start you multiprocessors */
  init();

  /* Parse input files */
  while(optind < argc)
  {
    tmp = optind++;
    printf("Reading '%s', ",argv[tmp]);
    fflush(stdout);

    memset(&image,0,sizeof(image));
    image = read_yuv(argv[tmp]);

	printf("Encoding, ");
    encode(image, tmp);

    free(image->Y);
    free(image->U);
    free(image->V);

    free(image);
	printf("Done!\n");
  }

  cleanup();
  fclose(outfile);

  return EXIT_SUCCESS;
}
