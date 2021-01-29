#ifndef _KERNELS_GAINCOMPENSATION_CU_
#define _KERNELS_GAINCOMPENSATION_CU_

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

__global__ void GainCompensation_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, float gain)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	uchar4 inrgba;
	//surf2Dread(&inrgba, inputSurf, ix * sizeof(uchar4), iy);

	float4 finrgba = tex2D<float4>(inputTex, __fdividef(ix , width), __fdividef(iy , height));

	inrgba.x = __fmul_rn(finrgba.x , 255.f);
	inrgba.y = __fmul_rn(finrgba.y , 255.f);
	inrgba.z = __fmul_rn(finrgba.z , 255.f);
	inrgba.w = __fmul_rn(finrgba.w , 255.f);

	int y = ((66 * inrgba.x + 129 * inrgba.y + 25 * inrgba.z + 128) >> 8) + 16;
	int u = ((-38 * inrgba.x - 74 * inrgba.y + 112 * inrgba.z + 128) >> 8) + 128;
	int v = ((112 * inrgba.x - 94 * inrgba.y - 18 * inrgba.z + 128) >> 8) + 128;

	float y_ = __fmul_rn(1.1643f, (__fdividef(y , 255.0f) - 0.0625f));
	float u_ = __fdividef(u , 255.0f) - 0.5f;
	float v_ = __fdividef(v , 255.0f) - 0.5f;
	y_ = __fmul_rn(y_ , gain);

	float r_ = __fmaf_rn(1.5958f, v_, y_ );
	float g_ = y_ - __fmul_rn(0.39173f, u_) - __fmul_rn(0.81290f, v_);
	float b_ = __fmaf_rn(2.017f, u_, y_ );

	r_ = fminf(r_, 1.0f);
	g_ = fminf(g_, 1.0f);
	b_ = fminf(b_, 1.0f);

	unsigned char r = __fmul_rn(r_ , 255);
	unsigned char g = __fmul_rn(g_ , 255);
	unsigned char b = __fmul_rn(b_ , 255);

	uchar4 outSample;
	outSample.x = r;
	outSample.y = g;
	outSample.z = b;
	outSample.w = inrgba.w;
	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C"
void runGainCompensation_Kernel(cudaSurfaceObject_t outputSurf, cudaSurfaceObject_t inputSurf, int width, int height, float gain)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	GainCompensation_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputSurf, width, height, gain);
}

#endif //_KERNELS_GAINCOMPENSATION_CU_