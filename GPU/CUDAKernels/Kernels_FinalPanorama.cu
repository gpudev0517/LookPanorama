

#ifndef _KERNELS_FINALPANORAMA_CU_
#define _KERNELS_FINALPANORAMA_CU_

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

#define LeftView 0
#define RightView 1


__global__ void Panorama_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex1, cudaTextureObject_t inputTex2, int width, int height,
	bool isStereo, bool isOutput)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;


	uint2 uv;
	uv.x = ix;
	uv.y = iy;

	bool isRight = false;
	if (isStereo)
	{
		if (__fmul_rn(uv.y , 2)< height)
		{
			uv.y = __fmul_rn(uv.y, 2);
		}
		else
		{
			uv.y = __fmul_rn((uv.y - __fdividef(height, 2.f)) , 2);
			isRight = true;
		}
	}

	if (isOutput)
		uv.y = height - uv.y - 1;

	//uchar4 src;
	float4 src;
	if (isRight)
	{
		src = tex2D<float4>(inputTex2, __fdividef(uv.x, width), __fdividef(uv.y , height));
		//surf2Dread(&src, inputSurf2, uv.x * sizeof(uchar4), uv.y);
	}
	else
	{
		src = tex2D<float4>(inputTex1, __fdividef(uv.x, width), __fdividef(uv.y, height));
		//surf2Dread(&src, inputSurf1, uv.x * sizeof(uchar4), uv.y);
	}
	uchar4 dst;
	dst.x = __fmul_rn(src.x , 255.f);
	dst.y = __fmul_rn(src.y , 255.f);
	dst.z = __fmul_rn(src.z , 255.f);
	dst.w = __fmul_rn(src.w , 255.f);
	surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C" void runPanorama_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex1, cudaTextureObject_t inputTex2, int width, int height, 
	bool isStereo, bool isOutput)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Panorama_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex1, inputTex2, width, height, isStereo, isOutput);
}

__global__ void FinalPanorama_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;


	int yuvIndex = 0;
	float2 pixTex;

	/*float x = (float)ix / (float)width;
	float y = (float)iy / (float)height;*/
	float x = __fdividef(ix, width);
	float y = __fdividef(iy, height);
	if (__fmul_rn(y , 3.0f) <= 2.0f)
	{
		yuvIndex = 0;
		pixTex.y = __fmul_rn(__fdividef(y, 2.0f) , 3.f);
		pixTex.x = x;
	}
	else
	{
		pixTex.y = __fmul_rn((y - 0.66667f) , 3.f);
		if (x <= 0.5f)
		{
			yuvIndex = 1;
			pixTex.x = __fmul_rn(x , 2.f);
		}
		else
		{
			yuvIndex = 2;
			pixTex.x = __fmul_rn((x - 0.5f) , 2.f);
		}
	}
	pixTex.x = __saturatef(pixTex.x);
	pixTex.y = __saturatef(pixTex.y);

	/*uchar4 color;
	surf2Dread(&color, inputSurf, (int)(pixTex.x * (float)(width - 1) ) * sizeof(uchar4), (int)(pixTex.y * (float)(height * 2 / 3 - 1)));*/
	float4 color = tex2D<float4>(inputTex, pixTex.x, pixTex.y);
	

	float b = __fmul_rn(color.x , 255.f);
	float g = __fmul_rn(color.y , 255.f);
	float r = __fmul_rn(color.z , 255.f);

	int value;

	if (yuvIndex == 0)
		value = (int)(__fmul_rn(0.257f, r) + __fmul_rn(0.504f, g) + __fmul_rn(0.098f , b)) + 16;
	else if (yuvIndex == 1)
		value = (int)(__fmul_rn(0.439f, r) - __fmul_rn(0.368f, g) - __fmul_rn(0.071f , b)) + 128;
	else
		value = (int)(__fmul_rn(-0.148f, r) - __fmul_rn(0.291f, g) + __fmul_rn(0.439f, b)) + 128;
	if (value < 0)
		value = 0;
	else if (value > 255)
		value = 255;

	uchar1 dst;
	dst.x = value;
	surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
}

extern "C" void runFinalPanorama_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height)
{

	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	FinalPanorama_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height);
}

#endif //_KERNELS_FINALPANORAMA_CU_