
#ifdef USE_CUDA

#include "CUDABoundary.h"
#include "define.h"
#include "common.h"


extern "C" void runBoundary_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t *inputTexs, int width, int height, int viewCnt);

CUDABoundary::CUDABoundary(QObject *parent) : GPUBoundary(parent)
{
}

CUDABoundary::~CUDABoundary()
{
}

void CUDABoundary::initialize(int panoWidth, int panoHeight, int viewCount)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	this->m_viewCount = viewCount;
	//this->m_viewIdx = viewIdx;
	
	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
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


void CUDABoundary::render(GPUResourceHandle *fbos)
{
	runBoundary_Kernel(m_cudaTargetSurface, fbos, panoramaWidth, panoramaHeight, m_viewCount);

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_Grayscale8);
	img.save(QString("boundary.png"));
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
}

#endif //USE_CUDA