

#ifndef _KERNELS_PANORAMAPOSTPROCESSING_CU_
#define _KERNELS_PANORAMAPOSTPROCESSING_CU_

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


__global__ void PanoramaPostProcessing_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, cudaSurfaceObject_t lutInpuSurf, cudaTextureObject_t seamInputTex,
	int width, int height, float3 ctLightColor, float *placeMat, bool seamOn)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float2 srcUV;
	srcUV.x = __fdividef(ix, width);
	srcUV.y = __fdividef(iy, height);
	float2 dstUV;
	XYnToDstXYn(srcUV, dstUV, placeMat);
	dstUV.x = fminf(dstUV.x, 1.f);
	dstUV.y = fminf(dstUV.y, 1.f);

	bool isSeam = false;
	if (seamOn){
		/*uchar4 seam;
		surf2Dread(&seam, seamInputSurf, (int)(dstUV.x * (float)(width - 1.0f)) * sizeof(uchar4), (int)(dstUV.y * (float)(height - 1.0f)));*/

		float1 seam = tex2D<float1>(seamInputTex, dstUV.x, dstUV.y);

		if (seam.x != 0.f){
			isSeam = true;
		}
	}

	if (isSeam == false){
		/*uchar4 cSrc;
		surf2Dread(&cSrc, inputSurf, (int)(dstUV.x * (float)(width - 1.0f)) * sizeof(uchar4), (int)(dstUV.y * (float)(height - 1.0f)));*/

		float4 fcSrc = tex2D<float4>(inputTex, dstUV.x, dstUV.y);
		uchar4 cSrc;
		cSrc.x = __fmul_rn(fcSrc.x , 255.f);
		cSrc.y = __fmul_rn(fcSrc.y , 255.f);
		cSrc.z = __fmul_rn(fcSrc.z , 255.f);
		cSrc.w = __fmul_rn(fcSrc.w , 255.f);
		uchar3 tmp = calcLut(lutInpuSurf, cSrc);
		tmp.x = __fmul_rn(tmp.x, ctLightColor.x);
		tmp.y = __fmul_rn(tmp.y, ctLightColor.y);
		tmp.z = __fmul_rn(tmp.z, ctLightColor.z);
		uchar4 dst;
		dst.x = fminf(tmp.x, 255);
		dst.y = fminf(tmp.y, 255);
		dst.z = fminf(tmp.z, 255);
		dst.w = cSrc.w;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);

	}
	else{
		uchar4 dst;
		dst.x = 255;
		dst.y = 0;
		dst.z = 0;
		dst.w = 255;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}

}

extern "C" void runPanoramaPostProcessing_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, cudaSurfaceObject_t lutInpuSurf, cudaTextureObject_t seamInputTex, 
	int width, int height, float3 ctLightColor, float *placeMat, bool seamOn)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	PanoramaPostProcessing_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, lutInpuSurf, seamInputTex, width, height, ctLightColor, placeMat, seamOn);
}



#endif //_KERNELS_PANORAMAPOSTPROCESSING_CU_