
#pragma once

#include <stdint.h>
#include <stdio.h>

#include <QImage>
#include "ImageBuffer.h"

enum eHCompression
{
	H_COMPRESSION_NONE = 0,	// Uncompressed sequence
	H_COMPRESSION_JPEG = 1,	// JPEG compressed images
	H_COMPRESSION_RLE = 2,	// RLE compressed images (stp3 algo)
	H_COMPRESSION_HUFFMAN = 3,	// HUFFMAN compressed images (stp3 algo)
	H_COMPRESSION_LZ = 4,	// LZ compressed images (stp3 algo)
	H_COMPRESSION_RLE_FAST = 5,	// RLE Fast compressed images (ippwrapper)
	H_COMPRESSION_HUFFMAN_FAST = 6,	// HUFFMAN Fast compressed images (ippwrapper
	H_COMPRESSION_LZ_FAST = 7,	// LZ Fast compressed images (ippwrapper)
	H_COMPRESSION_H264 = 8,	// H264 compressed images
	H_COMPRESSION_WAVELET = 9,	// Wavelet compressed images
	H_COMPRESSION_CREATEC_FLAT_1D = 10,	// Flat model predictor using 1 sample for Createc Bayer compressed images
	H_COMPRESSION_CREATEC_FLAT_2D = 11,	// Flat model predictor using 2 samples for Createc Bayer compressed images
	H_COMPRESSION_CREATEC_FLAT_BLUE_VECTOR_1D = 12, // Gradient predictor using the blue channel for Createc Bayer compressed 
	H_COMPRESSION_CREATEC_GREEN_AVERAGE = 13,	// Gradient predictor using the green channel average for Createc Bayer compressed 
	H_COMPRESSION_CREATEC_2D_LOSSY = 14,	// Flat_2D but with a reduced symbol set, Lossy codec with a maximum error of 1 for Createc Bayer compressed 
	H_COMPRESSION_CREATEC_FLAT_2D_SSE = 15,	// SSE2 implementation of Flat_2D for Createc Bayer compressed images
	H_COMPRESSION_CINEFORM = 16, // CINEFORM Codec
	H_COMPRESSION_JPEG_FV = 17  // Fast Video Compression JPEG
};


enum eHImageFormat
{
	H_IMAGE_UNKNOWN = 0,	// Unknown format
	H_IMAGE_MONO = 100,	// Monochrome Image (LSB)
	H_IMAGE_MONO_BAYER = 101,	// Raw Bayer Image (treated as H_IMAGE_MONO)
	H_IMAGE_BGR = 200,	// BGR Color Image
	H_IMAGE_PLANAR = 300,	// Planar Color Image
	H_IMAGE_RGB = 400,	// RGB Color Image
	H_IMAGE_BGRx = 500,	// BGRx Color Image
	H_IMAGE_YUV422 = 600,	// YUV422
	H_IMAGE_YUV422_20 = 610,
	H_IMAGE_UVY422 = 700,	// UVY422
	H_IMAGE_UVY411 = 800,	// UVY411
	H_IMAGE_UVY444 = 900,	// UVY444
	H_IMAGE_BGR555_PACKED = 905,	// PhynxRGB
	H_IMAGE_BGR565_PACKED = 906,
	H_IMAGE_MONO_MSB = 112,	// Only for > 8 bit per pixel, MSB align litle endian 10 bit: JIHGFEDC BA000000 
	H_IMAGE_MONO_BAYER_MSB = 113,	// Only for > 8 bit per pixel, MSB align 
	H_IMAGE_MONO_MSB_SWAP = 114,	// Only for > 8 bit per pixel, MSB align big endian 10 bit: BA000000 JIHGFEDC
	H_IMAGE_MONO_BAYER_MSB_SWAP = 115,	// Only for > 8 bit per pixel, MSB align
	H_IMAGE_BGR10_PPACKED = 123,	// Only for 10 bit per pixel, LSB align
	H_IMAGE_BGR10_PPACKED_PHOENIX = 124,	// Only for 10 bit per pixel, LSB align, RRRRRRRR RR00GGGG GGGGGGBB BBBBBBBB
	H_IMAGE_RGB10_PPACKED_PHOENIX = 125,	// Only for 10 bit per pixel, LSB align, BBBBBBBB BB00GGGG GGGGGGRR RRRRRRRR
	H_IMAGE_MONO_PPACKED = 131,	// Only for > 8 bit per pixel, MSB 
	H_IMAGE_MONO_BAYER_PPACKED = 132,	// Only for > 8 bit per pixel, MSB align
	H_IMAGE_MONO_PPACKED_8448 = 133,	// Only for > 8 bit per pixel, MSB align
	H_IMAGE_MONO_BAYER_PPACKED_8448 = 134,	// Only for > 8 bit per pixel, MSB align
	H_IMAGE_GVSP_BGR10V1_PACKED = 135,	// Only for 10 bit per pixel(From Gige Vision) BBBBBBBB GGGGGGGG RRRRRRRR 00BBGGRR
	H_IMAGE_GVSP_BGR10V2_PACKED = 136,	// Only for 10 bit per pixel(From Gige Vision)00BBBBBB BBGGGGGG GGGGRRRR 
	H_IMAGE_BASLER_VENDOR_SPECIFIC = 1000,
	H_IMAGE_EURESYS_JPEG = 1001,
	H_IMAGE_ISG_JPEG = 1002
};


class ImageHandler
{
public:
	enum FileMode {
		READ = 0,
		WRITE = 1
	};

	ImageHandler()
	{

	}
	~ImageHandler()
	{

	}

	virtual bool open(std::string fileName, int mode, int& xsize, int& ysize, float& frameRate, int mCompression = H_COMPRESSION_NONE,
		int compressionQuality = 100) = 0;
	virtual bool close() = 0;
	virtual bool read(int frameNum, AlignedImage& matFrame, QString fileName = "") = 0;
	virtual bool write(int frameNum, QImage& matFrame, int deviceNum = 0) = 0;

	int getMsToWait()
	{
		return msToWait;
	}
protected:

	FileMode mMode;
	int msToWait;

};
