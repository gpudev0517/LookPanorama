
#ifdef USE_CUDA
#include "CUDAFeathering.h"
#include "define.h"
#include "common.h"

extern "C" void runFeathering_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *colorInputTex, cudaTextureObject_t *weightInputTex, int width, int height, int viewCnt);

CUDAFeathering::CUDAFeathering(QObject *parent) : GPUFeathering(parent)
{
}

CUDAFeathering::~CUDAFeathering()
{
}

void CUDAFeathering::initialize(int panoWidth, int panoHeight, int viewCount)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	this->m_viewCount = viewCount;
	
	

	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&m_cudaTargetArray, &channelFormat, panoramaWidth, panoramaHeight, cudaArraySurfaceLoadStore);


	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);

	m_initialized = true;
}


void CUDAFeathering::render(GPUResourceHandle *fbos, GPUResourceHandle *weights, int compositeID)
{
	runFeathering_Kernel(m_cudaTargetSurface, fbos, weights, panoramaWidth, panoramaHeight, m_viewCount);

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_RGBA8888);
	img.save(QString("Fethering_") + QString::number(compositeID) + ".png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
}

#endif //USE_CUDA