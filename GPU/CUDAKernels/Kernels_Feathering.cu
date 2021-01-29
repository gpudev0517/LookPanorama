

#ifndef _KERNELS_FEATHERING_CU_
#define _KERNELS_FEATHERING_CU_

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


__global__ void Feathering_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *colorInputTex, cudaTextureObject_t *weightInputTex, int width, int height, int viewCnt)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float2 tex = make_float2(__fdividef(ix, width), __fdividef(iy, height));

	float ws = 0.0f;
	float3 csum;
	csum.x = 0;
	csum.y = 0;
	csum.z = 0;
	for (int i = 0; i < viewCnt; i++)
	{
		float4 src = getCameraColor(tex, colorInputTex[i], weightInputTex[i]);
		csum.x = __fmaf_rn(src.x, src.w, csum.x );
		csum.y = __fmaf_rn(src.y, src.w, csum.y );
		csum.z = __fmaf_rn(src.z, src.w, csum.z );
		ws = ws + src.w;
	}

	if (ws != 0.0f)
	{
		csum.x = __fdividef(csum.x , ws);
		csum.y = __fdividef(csum.y , ws);
		csum.z = __fdividef(csum.z , ws);
	}
	
	uchar4 dst;
	dst.x = __fmul_rn(csum.x , 255.f);
	dst.y = __fmul_rn(csum.y , 255.f);
	dst.z = __fmul_rn(csum.z , 255.f);
	dst.w = __fmul_rn(ws , 255.f);
	surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
}

extern "C" void runFeathering_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *colorInputTex, cudaTextureObject_t *weightInputTex, int width, int height, int viewCnt)
{
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Feathering_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, colorInputTex, weightInputTex, width, height, viewCnt);
}



#endif //_KERNELS_FEATHERING_CU_