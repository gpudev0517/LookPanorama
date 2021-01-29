#ifndef _KERNELS_COLORCVT_CU_
#define _KERNELS_COLORCVT_CU_

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#include <driver_functions.h>

#include "CudaCommon.cuh"

__global__ void ColorCvt_YUV2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, float width, float height, float bytesPerLine)
{
	
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float x = ix;
	float y = height - iy - 1;

	float texYx = x;
	float texYy = y;
	float texUx = __fdividef(x , 2.0f);
	float texUy = height + __fdividef(y , 4.0f);
	float texVx = __fdividef(x , 2.f);
	float texVy = 1.25f * height  + __fdividef(y , 4.f);

	if (fmodf(y, 2.f) == 1)
	{
		texUx += __fmul_rn(0.5f , bytesPerLine);
		texVx += __fmul_rn(0.5f , bytesPerLine);
	}

	float1 cy_ = tex2D<float1>(inputTex, __fdividef(texYx, bytesPerLine), __fdividef(__fdividef(texYy, height), 1.5f));
	float1 cu = tex2D<float1>(inputTex, __fdividef(texUx, bytesPerLine), __fdividef(__fdividef(texUy, height), 1.5f));
	float1 cv = tex2D<float1>(inputTex, __fdividef(texVx, bytesPerLine), __fdividef(__fdividef(texVy, height), 1.5f));

	float y_ = __fmul_rn(1.1643f, __fmaf_rn(cy_.x , 255.f, -15.9375f));
	float u = __fmaf_rn(cu.x , 255.f , -127.5f);
	float v = __fmaf_rn(cv.x , 255.f , - 127.5f);

	uchar4 outSample;
	outSample.x = fminf(__fmaf_rn(1.5958f, v, y_), 255.f);
	outSample.y = fminf(y_ - __fmul_rn(0.39173f, u) - __fmul_rn(0.81290f, v), 255.f);
	outSample.z = fminf(__fmaf_rn(2.017f, u, y_ ), 255.f);
	outSample.w = 255;
	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C" 
void runColorCvt_YUV2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	ColorCvt_YUV2RGBA_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, byteperline);
}



__global__ void ColorCvt_YUV4222RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, float width, float height, float bytesPerLine)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float4 fyuyv = tex2D<float4>(inputTex, __fdividef(__fmul_rn(ix, 2.f), bytesPerLine), 1.f - __fdividef(iy, height));

	/*uchar4 yuyv;
	surf2Dread(&yuyv, inputTex, ix / 2 * sizeof(uchar4), height - iy - 1);

	float4 fyuyv;
	fyuyv.x = yuyv.x / 255.f;
	fyuyv.y = yuyv.y / 255.f;
	fyuyv.z = yuyv.z / 255.f;
	fyuyv.w = yuyv.w / 255.f;
*/
	float u = fyuyv.y;
	float v = fyuyv.w;

	float y = 0.f;
	if (fmodf(ix, 2.f) == 0.f)
	{
		y = fyuyv.x;
		
	}
	else
	{
		y = fyuyv.z;
	}

	y = __fmul_rn(1.1643f, (y - 0.0625f));
	u = u - 0.5f;
	v = v - 0.5f;

	float r = __fmaf_rn(1.5958f, v, y );
	float g = y - __fmul_rn(0.39173f, u) - __fmul_rn(0.81290f, v);
	float b = __fmaf_rn(2.017f, u, y);

	r = __fmul_rn(255.f,  r);
	g = __fmul_rn(255.f , g);
	b = __fmul_rn(255.f , b);

	uchar4 outSample;
	outSample.x = fminf(r, 255.f);
	outSample.y = fminf(g, 255.f);
	outSample.z = fminf(b, 255.f);
	outSample.w = 255;

	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C"
void runColorCvt_YUV4222RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	ColorCvt_YUV4222RGBA_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, byteperline);
}




__global__ void ColorCvt_YUVJ422P2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, float width, float height, float bytesPerLine)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float x = ix;
	float y = height - iy - 1;

	float xhalf = __fdividef(x, 2.0f);
	float2 texY = make_float2(__fdividef(x, bytesPerLine), 0.0f + __fdividef(__fdividef(y, height), 2.0f));
	float2 texU = make_float2(__fdividef(xhalf, bytesPerLine), 0.5f + __fdividef(__fdividef(y, height), 4.0f));
	float2 texV = make_float2(__fdividef(xhalf, bytesPerLine), 0.75f + __fdividef(__fdividef(y, height), 4.0f));

	if (fmodf(y, 2.f) == 1)
	{
		texU.x += 0.5f;
		texV.x += 0.5f;
	}

	float y_ = tex2D<float1>(inputTex, texY.x, texY.y).x;
	float u = tex2D<float1>(inputTex, texU.x, texU.y).x;
	float v = tex2D<float1>(inputTex, texV.x, texV.y).x;

	u -= 0.5f;
	v -= 0.5f;

	uchar4 outSample;
	outSample.x = __fmul_rn(__fmaf_rn(1.403f, v, y_), 255.f);
	outSample.y = __fmul_rn(y_ - __fmul_rn(0.344f, u) - __fmul_rn(0.714f, v), 255.f);
	outSample.z = __fmul_rn(__fmaf_rn(1.770f, u, y_), 255.f);
	outSample.w = 255;
	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C"
void runColorCvt_YUVJ422P2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	ColorCvt_YUVJ422P2RGBA_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, byteperline);
}


__global__ void ColorCvt_BGR02RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, float width, float height, float bytesPerLine)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float1 inb = tex2D<float1>(inputTex, __fdividef(__fmul_rn(4, ix), bytesPerLine), 1.f - __fdividef(iy, height));
	float1 ing = tex2D<float1>(inputTex, __fdividef(__fmul_rn(4, ix) + 1, bytesPerLine), 1.f - __fdividef(iy, height));
	float1 inr = tex2D<float1>(inputTex, __fdividef(__fmul_rn(4, ix) + 2, bytesPerLine), 1.f - __fdividef(iy, height));

/*
	uchar1 inr;
	surf2Dread(&inr, inputSurf, 3 * ix * sizeof(uchar1), height - iy - 1);
	uchar1 ing;
	surf2Dread(&ing, inputSurf, (3 * ix + 1) * sizeof(uchar1), height - iy - 1);
	uchar1 inb;
	surf2Dread(&inb, inputSurf, (3 * ix + 2) * sizeof(uchar1), height - iy - 1);*/

	uchar4 outSample;
	outSample.x = __fmul_rn(inr.x , 255.f);
	outSample.y = __fmul_rn(ing.x , 255.f);
	outSample.z = __fmul_rn(inb.x , 255.f);
	outSample.w = 255;

	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}


extern "C"
void runColorCvt_BGR02RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	ColorCvt_BGR02RGBA_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, byteperline);
}


__global__ void ColorCvt_RGB2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, float width, float height, float bytesPerLine)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float1 inr = tex2D<float1>(inputTex, __fdividef(__fmul_rn(3, ix), bytesPerLine), 1.f - __fdividef(iy, height));
	float1 ing = tex2D<float1>(inputTex, __fdividef(__fmul_rn(3, ix) + 1, bytesPerLine), 1.f - __fdividef(iy, height));
	float1 inb = tex2D<float1>(inputTex, __fdividef(__fmul_rn(3, ix) + 2, bytesPerLine), 1.f - __fdividef(iy, height));

/*
	uchar1 inr;
	surf2Dread(&inr, inputSurf, 3 * ix * sizeof(uchar1), height - iy - 1);
	uchar1 ing;
	surf2Dread(&ing, inputSurf, (3 * ix + 1) * sizeof(uchar1), height - iy - 1);
	uchar1 inb;
	surf2Dread(&inb, inputSurf, (3 * ix + 2) * sizeof(uchar1), height - iy - 1);*/

	uchar4 outSample;
	outSample.x = __fmul_rn(inr.x , 255.f);
	outSample.y = __fmul_rn(ing.x , 255.f);
	outSample.z = __fmul_rn(inb.x , 255.f);
	outSample.w = 255;

	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}


extern "C"
void runColorCvt_RGB2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	ColorCvt_RGB2RGBA_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, byteperline);
}

__global__ void ColorCvt_RGBA2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, float width, float height, float bytesPerLine)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	//uchar4 inrgba;
	//surf2Dread(&inrgba, inputSurf, ix * sizeof(uchar4), height - iy - 1);

	float4 inrgba = tex2D<float4>(inputTex, __fdividef(ix, bytesPerLine), 1.f - __fdividef(iy, height));

	uchar4 outSample;
	outSample.x = __fmul_rn(inrgba.x, 255.f);
	outSample.y = __fmul_rn(inrgba.y, 255.f);
	outSample.z = __fmul_rn(inrgba.z, 255.f);
	outSample.w = __fmul_rn(inrgba.w, 255.f);
	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C"
void runColorCvt_RGBA2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	ColorCvt_RGBA2RGBA_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, byteperline);
}

#endif //_KERNELS_COLORCVT_CU_