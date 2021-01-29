

#ifndef _KERNELS_UNWARP_CU_
#define _KERNELS_UNWARP_CU_

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


__global__ void Unwarp_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int renderMode, int lens, float imageWidth, float imageHeight, float offset_x, float offset_y,
	float k1, float k2, float k3, float FoV, float FoVY, float *cP)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	

	float cx = __fdividef(imageWidth , 2.f);
	float cy = __fdividef(imageHeight , 2.f);


	float2 camera;
	float2 texc;
	texc.x = __fdividef(ix , width);
	texc.y = __fdividef(iy , height);

	float alpha = XYnToLocal(texc, 1.0f, camera, lens, cx, cy, offset_x, offset_y, k1, k2, k3, FoV, FoVY, cP);

	if (alpha != 0.0)
	{
		if (renderMode == RenderMode_Color){
			uchar4 cSrc;
			//surf2Dread(&cSrc, inputSurf, (int)camera.x * sizeof(uchar4), (int)camera.y);

			float4 fcSrc = tex2D<float4>(inputTex, __fdividef(camera.x, imageWidth), __fdividef(camera.y , imageHeight));
			
			cSrc.x = __fmul_rn(fcSrc.x , 255.f);
			cSrc.y = __fmul_rn(fcSrc.y , 255.f);
			cSrc.z = __fmul_rn(fcSrc.z , 255.f);
			cSrc.w = fminf(__fmul_rn(alpha , 255.f), 255.f);

			surf2Dwrite(cSrc, outputSurf, ix * sizeof(uchar4), iy);

		}
		else{
			uchar4 cSrc;
			cSrc.x = fminf(__fmul_rn(alpha , 255.f), 255.f);
			cSrc.y = 0;
			cSrc.z = 0;
			cSrc.w = fminf(__fmul_rn(alpha , 255.f), 255.f);
			surf2Dwrite(cSrc, outputSurf, ix * sizeof(uchar4), iy);
		}
	}
	else
	{	//discard;
		uchar4 cSrc;
		cSrc.x = 0;
		cSrc.y = 0;
		cSrc.z = 0;
		cSrc.w = 0;
		surf2Dwrite(cSrc, outputSurf, ix * sizeof(uchar4), iy);
	}

}

extern "C" void runUnwarp_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int renderMode, int lens, float imageWidth, float imageHeight, 
	float offset_x, float offset_y,	float k1, float k2, float k3, float FoV, float FoVY, float *cP)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Unwarp_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, renderMode, lens, imageWidth, imageHeight, offset_x, offset_y, k1, k2, k3, FoV, FoVY, cP);
}



#endif //_KERNELS_UNWARP_CU_