

#ifndef _KERNELS_SEAM_CU_
#define _KERNELS_SEAM_CU_

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


__global__ void SeamRegion_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int viewCnt, int viewIdx)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	uchar1 dst;
	dst.x = 0;
	surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);


	if (viewIdx == 0)
	{
		/*uchar1 src0;
		surf2Dread(&src0, inputSurf, ix * sizeof(uchar1), iy);*/

		float1 src0 = tex2D<float1>(inputTex, __fdividef(ix, width), __fdividef(iy, height));

		dst.x = __fmul_rn(src0.x , 255.f);
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
	}
	else
	{

		if (isActive(inputTex, make_float2(__fdividef(ix, width), __fdividef(iy, height)), viewIdx - 1, viewCnt) == true){
			dst.x = 255;
			surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
		}
	}
}

extern "C" void runSeamRegion_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int viewCnt, int viewIdx)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	SeamRegion_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height, viewCnt, viewIdx);
}




__global__ void SeamMask_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	if (ix == 0 || ix == width - 1 || iy == 0 || iy == height - 1){
		uchar1 dst;
		dst.x = 0;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
	}
	else{
		/*uchar4 src0;
		surf2Dread(&src0, inputSurf, ix * sizeof(uchar4), iy);
		uchar4 src1;
		surf2Dread(&src1, inputSurf, (ix + 1) * sizeof(uchar4), iy);
		uchar4 src2;
		surf2Dread(&src2, inputSurf, (ix - 1) * sizeof(uchar4), iy);
		uchar4 src3;
		surf2Dread(&src3, inputSurf, ix * sizeof(uchar4), iy - 1);
		uchar4 src4;
		surf2Dread(&src4, inputSurf, ix * sizeof(uchar4), iy + 1);*/

		float1 src0 = tex2D<float1>(inputTex, __fdividef(ix, width), __fdividef(iy, height));
		float1 src1 = tex2D<float1>(inputTex, __fdividef(ix+1, width), __fdividef(iy, height));
		float1 src2 = tex2D<float1>(inputTex, __fdividef(ix-1, width), __fdividef(iy, height));
		float1 src3 = tex2D<float1>(inputTex, __fdividef(ix, width), __fdividef(iy-1, height));
		float1 src4 = tex2D<float1>(inputTex, __fdividef(ix, width), __fdividef(iy-1, height));


		if ((src0.x == src1.x) && (src0.x == src2.x) && (src0.x == src3.x) && (src0.x == src4.x)){
			uchar1 dst;
			dst.x = 0;
			surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
		}
		else{
			uchar1 dst;
			dst.x = 255;
			surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
		}
	}
}

extern "C" void runSeamMask_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height)
{

	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	SeamMask_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, width, height);
}



__global__ void Seam_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, cudaTextureObject_t maskTex, int width, int height)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	
	/*uchar4 src;
	surf2Dread(&src, inputSurf, ix * sizeof(uchar4), iy);
	uchar4 maskSrc;
	surf2Dread(&maskSrc, maskSurf, ix * sizeof(uchar4), iy);*/

	float4 src = tex2D<float4>(inputTex, (float)ix / (float)width, (float)iy / (float)height);
	float4 maskSrc = tex2D<float4>(maskTex, (float)ix / (float)width, (float)iy / (float)height);

	if (maskSrc.x == 0.f){
		uchar4 dst;
		dst.x = src.x * 255.f;
		dst.y = src.y * 255.f;
		dst.z = src.z * 255.f;
		dst.w = src.w * 255.f;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
	else{
		uchar4 dst;
		dst.x = maskSrc.x * 255.f;
		dst.y = 0;
		dst.z = 0;
		dst.w = 255;
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
}

extern "C" void runSeam_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, cudaTextureObject_t maskTex, int width, int height)
{

	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Seam_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTex, maskTex, width, height);
}

#endif //_KERNELS_SEAM_CU_