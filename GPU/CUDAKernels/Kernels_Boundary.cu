

#ifndef _KERNELS_BOUDNDARY_CU_
#define _KERNELS_BOUDNDARY_CU_

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


__global__ void Boundary_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *inputTexs, int width, int height, int viewCnt)
{

	unsigned int ix = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

	if (ix >= width || iy >= height)
		return;

	uchar1 dst;
	dst.x = 0;
	surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);

	for (int currentIdx = 0; currentIdx < viewCnt; currentIdx++)
	{
		/*uchar1 src0;
		surf2Dread(&src0, inputSurf[currentIdx], ix * sizeof(uchar1), iy);*/

		float1 src0 = tex2D<float1>(inputTexs[currentIdx], __fdividef(ix, width), __fdividef(iy , height));

		bool isPrime = true;
		for (int i = 0; i < viewCnt; i++)
		{
			if (i != currentIdx)
			{
				/*uchar1 src1;
				surf2Dread(&src1, inputSurf[i], ix * sizeof(uchar1), iy);*/

				float1 src1 = tex2D<float1>(inputTexs[i], __fdividef(ix, width), __fdividef(iy, height));
				
				if (src1.x > src0.x)
				{
					isPrime = false;
					break;
				}
				
			}
		}
		if (isPrime)
		{
			dst.x = __fmul_rn(255.f , __fdividef(currentIdx + 1.f, viewCnt));
			surf2Dwrite(dst, outputSurf, ix * sizeof(uchar1), iy);
		}
	}
}

extern "C" void runBoundary_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *inputTexs, int width, int height, int viewCnt)
{
	
	dim3 dimBlock(PANO_BLOCK_NUM, PANO_BLOCK_NUM, 1);
	dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y, 1);
	Boundary_Kernel << < dimGrid, dimBlock, 0, g_CurStream >> >(outputSurf, inputTexs, width, height, viewCnt);
}



#endif //_KERNELS_BOUDNDARY_CU_