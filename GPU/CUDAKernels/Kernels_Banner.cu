

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


__global__ void Bill_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t bgInputTex, cudaTextureObject_t bannerInputTex, int width, int height,
	float *bannerPaiPlane, float bannerPaiZdotOrg, float *bannerHomography, int widthPixels, int heightPixels, int imgWidth, int imgHeight)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;


	float2 vt;
	int cornerCount = 9;
	float2 offset[9];
	float pxWidth = 0.5f;
	offset[0].x = 0;
	offset[0].y = 0;
	offset[1].x = __fdividef(pxWidth , widthPixels);
	offset[1].y = 0;
	offset[2].x = 0;
	offset[2].y = __fdividef(pxWidth , heightPixels);
	offset[3].x = __fdividef(pxWidth , widthPixels);
	offset[3].y = __fdividef(pxWidth , heightPixels);
	offset[4].x = __fdividef(-pxWidth , widthPixels);
	offset[4].y = 0;
	offset[5].x = 0;
	offset[5].y = __fdividef(-pxWidth , heightPixels);
	offset[6].x = __fdividef(-pxWidth , widthPixels);
	offset[6].y = __fdividef(-pxWidth , heightPixels);
	offset[7].x = __fdividef(pxWidth , widthPixels);
	offset[7].y = __fdividef(-pxWidth , heightPixels);
	offset[8].x = __fdividef(-pxWidth , widthPixels);
	offset[8].y = __fdividef(pxWidth , heightPixels);
	bool isBanner[9];
	float2 vts[9];
	bool isInside = true;
	bool bannerFlag;
	float2 texc;
	texc.x = __fdividef(ix, width);
	texc.y = __fdividef(iy, height);
	for (int i = 0; i < cornerCount; i++)
	{
		float2 vt;
		float2 fIn;
		fIn.x = texc.x + offset[i].x;
		fIn.y = texc.y + offset[i].y;
		isBanner[i] = UVToBannerCoord(fIn, vt, bannerPaiPlane, bannerPaiZdotOrg, bannerHomography);
		vts[i] = vt;
		if (i == 0)
		{
			bannerFlag = isBanner[i];
		}
		else
		{
			if (bannerFlag != isBanner[i])
			{
				isInside = false;
			}
		}
	}
	if (isInside)
	{
		/*uchar4 tBackground;
		surf2Dread(&tBackground, bgInputSurf, ix * sizeof(uchar4), iy);*/

		float4 tBackground = tex2D<float4>(bgInputTex, __fdividef(ix, width), __fdividef(iy, height));

		if (isBanner[0] == true)
		{
			float2 vt = vts[0];
			/*uchar4 tBanner;
			surf2Dread(&tBanner, bannerInputSurf, (int)(vt.x * (float)(imgWidth-1)) * sizeof(uchar4), (int)(vt.y * (float)(imgHeight-1)));*/
			float4 tBanner = tex2D<float4>(bannerInputTex, vt.x, vt.y);
			float4 fdst = mix(tBackground, tBanner, tBanner.w);

			uchar4 dst;
			dst.x = __fmul_rn(fdst.x , 255.f); 
			dst.y = __fmul_rn(fdst.y , 255.f);
			dst.z = __fmul_rn(fdst.z , 255.f);
			dst.w = __fmul_rn(fdst.w , 255.f);
			surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
		}
		else
		{
			uchar4 dst;
			dst.x = __fmul_rn(tBackground.x , 255.f);
			dst.y = __fmul_rn(tBackground.y , 255.f);
			dst.z = __fmul_rn(tBackground.z , 255.f);
			dst.w = __fmul_rn(tBackground.w , 255.f);
			surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
		}
	}
	else
	{
		float4 color;
		color.x = 0;
		color.y = 0;
		color.z = 0;
		color.w = 0;
		for (int i = 0; i < cornerCount; i++)
		{
			//uchar4 singleColor;
			float4 singleColor;
			if (isBanner[i] == true)
			{
				singleColor = tex2D<float4>(bannerInputTex, vts[i].x, vts[i].y);
				//surf2Dread(&singleColor, bannerInputSurf, (int)(vts[i].x * (float)(imgWidth-1))*sizeof(uchar4), (int)(vts[i].y * (float)(imgHeight-1)));
			}
			else
			{
				/*int idx = (int)(ix + offset[i].x * (float)width);
				int idy = (int)(iy + offset[i].y * (float)height);*/

				float idx = (texc.x + offset[i].x );
				float idy = (texc.y + offset[i].y);
				singleColor = tex2D<float4>(bgInputTex, idx, idy);
				//surf2Dread(&singleColor, bgInputSurf, idx * sizeof(uchar4), idy);
			}
			color = mix(color, singleColor, __fdividef(singleColor.w , 9.0f));
		}

		uchar4 dst;
		dst.x = __fmul_rn(color.x , 255.f);
		dst.y = __fmul_rn(color.y , 255.f);
		dst.z = __fmul_rn(color.z , 255.f);
		dst.w = __fmul_rn(color.w , 255.f);
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
}

extern "C" void runBill_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t bgInputTex, cudaTextureObject_t bannerInputTex, int width, int height, 
	float *bannerPaiPlane, float bannerPaiZdotOrg, float *bannerHomography, int widthPixels, int heightPixels, int imgWidth, int imgHeight)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Bill_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, bgInputTex, bannerInputTex, width, height, bannerPaiPlane, bannerPaiZdotOrg, bannerHomography, widthPixels, heightPixels, imgWidth, imgHeight);
}

__global__ void BillClear_Kernel(cudaSurfaceObject_t outputSurf, int width, int height)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	uchar4 dst = make_uchar4(0, 0, 0, 0);
	surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	
}


extern "C" void runBillClear_Kernel(cudaSurfaceObject_t outputSurf, int width, int height)
{

	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	BillClear_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, width, height);
}


__global__ void Banner_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t bgInputTex, cudaTextureObject_t bannerInputTex, int width, int height)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;


	/*uchar4 bg;
	surf2Dread(&bg, bgInputSurf, ix * sizeof(uchar4), iy);
	uchar4 banner;
	surf2Dread(&banner, bannerInputSurf, ix * sizeof(uchar4), iy);*/

	float4 bg = tex2D<float4>(bgInputTex, __fdividef(ix, width), __fdividef(iy, height));
	float4 banner = tex2D<float4>(bannerInputTex, __fdividef(ix, width), __fdividef(iy, height));

	if (banner.w == 0.f)
	{
		uchar4 dst = make_uchar4(bg.x * 255.f, bg.y * 255.f, bg.z * 255.f, bg.w * 255.f);
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
	else
	{
		uchar4 dst;
		dst.x = __fmul_rn(255.f , (__fmul_rn(banner.x, banner.w) + __fmul_rn(bg.x, (1.f - banner.w))));
		dst.y = __fmul_rn(255.f , (__fmul_rn(banner.y, banner.w) + __fmul_rn(bg.y, (1.f - banner.w))));
		dst.z = __fmul_rn(255.f , (__fmul_rn(banner.z, banner.w) + __fmul_rn(bg.z, (1.f - banner.w))));
		dst.w = __fmul_rn(bg.w , 255.f);
		surf2Dwrite(dst, outputSurf, ix * sizeof(uchar4), iy);
	}
}

extern "C" void runBanner_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t bgInputTex, cudaTextureObject_t bannerInputTex, int width, int height)
{

	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Banner_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, bgInputTex, bannerInputTex, width, height);
}


#endif //_KERNELS_UNWARP_CU_