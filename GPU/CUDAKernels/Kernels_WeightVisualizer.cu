

#ifndef _KERNELS_WEIGHTVISUALIZER_CU_
#define _KERNELS_WEIGHTVISUALIZER_CU_

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


__global__ void WeightVisualizer_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *colorInputTex, cudaTextureObject_t *weightInputTex, int width, int height, int viewCnt,
	int currentIndex, int eyeIndex, int paintMode, int eyeMode)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	if (paintMode == 1){ //Paint Mode
		float3 viewActiveColor;
		viewActiveColor.x = 0.0f;
		viewActiveColor.y = 1.0f;
		viewActiveColor.z = 0.0f;
		float3 viewOtherColor;
		viewOtherColor.x = 0.5f;
		viewOtherColor.y = 0.5f;
		viewOtherColor.z = 0.5f;

		float ws = 0.0f;
		float3 csum;
		csum.x = 0;
		csum.y = 0;
		csum.z = 0;
		for (int i = 0; i < viewCnt; i++)
		{
			bool beyeIndex = eyeIndex == 0 ? false : true;
			float3 viewColor;
			if (eyeMode == 3){
				viewColor = (i == currentIndex) ? viewActiveColor : viewOtherColor;
			}
			else{
				bool isRightEye = (eyeMode == 2) ? true : false;
				viewColor = (beyeIndex == isRightEye && i == currentIndex) ? viewActiveColor : viewOtherColor;
			}

			float4 src = getCameraColor(make_float2(__fdividef(ix, width), __fdividef(iy, height)), colorInputTex[i], weightInputTex[i]);
			float weight = src.w;

			float3 csrc;
			if (i == currentIndex){
				if (eyeMode == 3)
				{
					csrc.x = 0.2f * src.x + viewColor.x * weight;
					csrc.y = 0.2f * src.y + viewColor.y * weight;
					csrc.z = 0.2f * src.z + viewColor.z * weight;
				}
				else{
					bool isRightEye = (eyeMode == 2) ? true : false;
					if (beyeIndex == isRightEye){
						csrc.x = 0.2f * src.x + viewColor.x * weight;
						csrc.y = 0.2f * src.y + viewColor.y * weight;
						csrc.z = 0.2f * src.z + viewColor.z * weight;
					}
					else{
						csrc.x = src.x * viewColor.x * weight;
						csrc.y = src.y * viewColor.y * weight;
						csrc.z = src.z * viewColor.z * weight;
					}
				}
			}
			else
			{
				csrc.x = src.x * viewColor.x * weight;
				csrc.y = src.y * viewColor.y * weight;
				csrc.z = src.z * viewColor.z * weight;
			}

			csum.x = csum.x + csrc.x; csum.y = csum.y + csrc.y; csum.z = csum.z + csrc.z;
			ws = ws + weight; //weight.r;
		}

		if (ws == 0.0f)
		{
			csum.x = 0;
			csum.y = 0;
			csum.z = 0;
		}

		csum.x =fminf(csum.x, 1.0f); csum.y = fminf(csum.y, 1.0f); csum.z = fminf(csum.z, 1.0f);

		uchar4 dst;
		dst.x = csum.x * 255.f;
		dst.y = csum.y * 255.f;
		dst.z = csum.z * 255.f;
		dst.w = 255;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
	else if (paintMode == 2){ //View Mode

		float ws = 0.0f;
		float3 csum;
		csum.x = 0;
		csum.y = 0;
		csum.z = 0;
		for (int i = 0; i < viewCnt; i++)
		{

			float4 src = getCameraColor(make_float2(__fdividef(ix, width), __fdividef(iy, height)), colorInputTex[i], weightInputTex[i]);

			
			float weight = src.w;

			csum.x = csum.x + src.x * weight; 
			csum.y = csum.y + src.y * weight; 
			csum.z = csum.z + src.z * weight;
			ws = ws + weight; //weight.r;
		}

		if (ws == 0.0f)
		{
			csum.x = 0;
			csum.y = 0;
			csum.z = 0;
		}
		else{
			csum.x = __fdividef(csum.x , ws); csum.y = __fdividef(csum.y , ws); csum.z = __fdividef(csum.z , ws);
		}

		csum.x = fminf(csum.x, 1.0f); csum.y = fminf(csum.y, 1.0f); csum.z = fminf(csum.z, 1.0f);
		
		uchar4 dst;
		dst.x = csum.x * 255.f;
		dst.y = csum.y * 255.f;
		dst.z = csum.z * 255.f;
		dst.w = ws * 255.f;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
	else if (paintMode == 3){  //Overlap Mode

		float ws = 0.0f;
		float3 csum;
		csum.x = 0.0f;
		csum.y = 0.0f;
		csum.z = 0.0f;
		int overlapIdx = 0;
		for (int i = 0; i < viewCnt; i++)
		{

			float4 src = getCameraColor(make_float2(__fdividef(ix, width), __fdividef(iy, height)), colorInputTex[i], weightInputTex[i]);
			float weight = src.w;

			if (src.w != 0.f){
				overlapIdx = overlapIdx + 1;
			}

			csum.x = csum.x + src.x * weight;
			csum.y = csum.y + src.y * weight;
			csum.z = csum.z + src.z * weight;
			ws = ws + weight; //weight.r;
		}

		if (ws == 0.0f)
		{
			csum.x = 0.0f;
			csum.y = 0.0f;
			csum.z = 0.0f;
		}
		else{
			csum.x = __fdividef(csum.x , ws); csum.y = __fdividef(csum.y , ws); csum.z = __fdividef(csum.z , ws);
		}
		

		if (overlapIdx > 1){
			csum.x = csum.x * 0.5; csum.y = csum.y * 0.5; csum.z = csum.z * 0.5;
		}

		csum.x = fminf(csum.x, 1.0f); csum.y = fminf(csum.y, 1.0f); csum.z = fminf(csum.z, 1.0f);

		uchar4 dst;
		dst.x = csum.x * 255.f;
		dst.y = csum.y * 255.f;
		dst.z = csum.z * 255.f;
		dst.w = ws * 255.f;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
}

extern "C" void runWeightVisualizer_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *colorInputTex, cudaTextureObject_t *weightInputTex, int width, int height, int viewCnt,
	int currentIndex, int eyeIndex, int paintMode, int eyeMode)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	WeightVisualizer_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, colorInputTex, weightInputTex, width, height, viewCnt, currentIndex, eyeIndex, paintMode, eyeMode);
}



#endif //_KERNELS_WEIGHTVISUALIZER_CU_