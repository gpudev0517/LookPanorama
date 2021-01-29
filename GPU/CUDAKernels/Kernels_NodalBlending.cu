

#ifndef _KERNELS_NODALBLENDING_CU_
#define _KERNELS_NODALBLENDING_CU_

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


__global__ void NodalBlending_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t forgroundTex, cudaTextureObject_t *bgColorTex, cudaTextureObject_t *bgWeightTex, int *bgWeightOn, int width, int height, int bgCnt)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;


	// Get foreground
	float4 fgColor = tex2D<float4>(forgroundTex, __fdividef(ix, width), __fdividef(iy, height));

	// Get background
	float3 bgColorSum;
	bgColorSum.x = 0;
	bgColorSum.y = 0;
	bgColorSum.z = 0;
	float bgWeightSum = 0.0f;
	for (int i = 0; i < bgCnt; i++)
	{
		float4 bgColor = tex2D<float4>(bgColorTex[i], __fdividef(ix, width), __fdividef(iy, height));
		float bgWeight = tex2D<float4>(bgWeightTex[i], __fdividef(ix, width), __fdividef(iy, height)).x;

		bgColorSum.x = bgColorSum.y + bgColor.x * bgWeight;
		bgColorSum.y = bgColorSum.y + bgColor.y * bgWeight;
		bgColorSum.z = bgColorSum.z + bgColor.z * bgWeight;
		bgWeightSum = bgWeightSum + bgWeight;
	}
	
	// Composite
	uchar4 result;
	result.x = 0;
	result.y = 0;
	result.z = 0;
	result.w = 0;
	float3 csum;
	
	csum.x = bgColorSum.x + fgColor.x * fgColor.w;
	csum.y = bgColorSum.y + fgColor.y * fgColor.w;
	csum.z = bgColorSum.z + fgColor.z * fgColor.w;
	float ws = bgWeightSum + fgColor.w;

	if (ws != 0.0f)
	{
		result.x = fminf(255.f * __fdividef(csum.x, ws), 255.f);
		result.y = fminf(255.f * __fdividef(csum.y, ws), 255.f);
		result.z = fminf(255.f * __fdividef(csum.z, ws), 255.f);
		result.w = ws;
	}

	surf2Dwrite(result, outputSurf, ix * sizeof(uchar4), iy);

}

extern "C" void runNodalBlending_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t forgroundTex, cudaTextureObject_t *bgColorTex, cudaTextureObject_t *bgWeightTex, int *bgWeightOn, int width, int height, int bgCnt)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	NodalBlending_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, forgroundTex, bgColorTex, bgWeightTex, bgWeightOn, width, height, bgCnt);
}



#endif //_KERNELS_NODALBLENDING_CU_