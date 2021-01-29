

#ifndef _KERNELS_WEIGHTMAP_CU_
#define _KERNELS_WEIGHTMAP_CU_

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


__global__ void CameraWeightMap_Kernel(cudaSurfaceObject_t outputSurf, int width, int height, float blendingFalloff,
	float fisheyeLensRadiusRatio1, float fisheyeLensRadiusRatio2, float xrad1, float xrad2, float yrad1, float yrad2, float blendCurveStart, int lens)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float2 camera;
	camera.x = (float)ix;
	camera.y = (float)iy;

	float radialDist2 = 1.0f - getRadialDistance2(camera, width, height, blendingFalloff, fisheyeLensRadiusRatio1, fisheyeLensRadiusRatio2, xrad1, xrad2, yrad1, yrad2, blendCurveStart, lens);;

	radialDist2 = fminf(__fmul_rn(255.f , radialDist2), 255.f);
	

	uchar1 outSample;
	outSample.x = radialDist2;
	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar1), iy);
}

extern "C" void runCameraWeightMap_Kernel(cudaSurfaceObject_t outputSurf, int width, int height, float blendFalloff,
	float fisheyeLensRadiusRatio1, float fisheyeLensRadiusRatio2, float xrad1, float xrad2, float yrad1, float yrad2, float blendCurveStart, int lens)
{
	dim3 dimBlock(CAM_BLOCK_NUM, CAM_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	CameraWeightMap_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, width, height, blendFalloff, fisheyeLensRadiusRatio1, fisheyeLensRadiusRatio2,
		xrad1, xrad2, yrad1, yrad2, blendCurveStart, lens);
}



__global__ void PanoramaWeightMap_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t weightTex, cudaTextureObject_t deltaWeightTex, int width, int height,
	int lens, float cx, float cy, float offset_x, float offset_y, float k1, float k2, float k3, float FoV, float FoVY, float *cP)
{
	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	float2 camera;
	float2 texc;
	texc.x = __fdividef(ix , width);
	texc.y = __fdividef(iy ,height);
	float alpha = XYnToLocal(texc, 1.0f, camera, lens, cx, cy, offset_x, offset_y, k1, k2, k3, FoV, FoVY, cP);

	uchar1 outSample;
	outSample.x = 0;
	surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar1), iy);

	if (alpha != 0.0f)
	{
		
		unsigned int idx = camera.x;
		unsigned int idy = camera.y;
		/*uchar1 inweight;
		surf2Dread(&inweight, weightSurf, idx * sizeof(uchar1), idy);
		float1 indeltaweight;
		surf2Dread(&indeltaweight, deltaWeightSurf, idx * sizeof(float1), idy);*/

		float1 inweight = tex2D<float1>(weightTex, __fdividef(__fdividef(idx, cx), 2.f), __fdividef(__fdividef(idy, cy), 2.f));
		float1 indeltaweight = tex2D<float1>(deltaWeightTex, __fdividef(__fdividef(idx, cx), 2.f), __fdividef(__fdividef(idy, cy), 2.f));

		float finalWeight = 0.f;

		finalWeight = inweight.x + indeltaweight.x - 0.5f;

		if (finalWeight < 0.0f)
			finalWeight = 0.f;
		if (finalWeight > 1.f)
			finalWeight = 1.f;

		outSample.x = __fmul_rn(finalWeight , 255.f);
		surf2Dwrite(outSample, outputSurf, ix * sizeof(uchar1), iy);
	}
}

extern "C" void runPanoramaWeightMap_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t weightTex, cudaTextureObject_t deltaWeightTex, int width, int height,
	int lens, float cx, float cy, float offset_x, float offset_y, float k1, float k2, float k3, float FoV, float FoVY, float *cP)
{
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	PanoramaWeightMap_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, weightTex, deltaWeightTex, width, height,
		lens, cx, cy, offset_x, offset_y, k1, k2, k3, FoV, FoVY, cP);
}

#endif //_KERNELS_WEIGHTMAP_CU_