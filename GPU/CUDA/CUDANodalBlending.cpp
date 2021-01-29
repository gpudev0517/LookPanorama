
#ifdef USE_CUDA

#include "CUDANodalBlending.h"
#include "define.h"
#include "common.h"
extern "C" void runNodalBlending_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t forgroundTex, cudaTextureObject_t *bgColorTex, cudaTextureObject_t *bgWeightTex, int *bgWeightOn, int width, int height, int bgCnt);


CUDANodalBlending::CUDANodalBlending(QObject *parent) : GPUNodalBlending(parent)
{
}

CUDANodalBlending::~CUDANodalBlending()
{
}


void CUDANodalBlending::initialize(int panoWidth, int panoHeight, int nodalCount, bool haveNodalMaskImage)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	this->nodalCameraCount = nodalCameraCount;
	this->haveNodalMaskImage = haveNodalMaskImage;

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


void CUDANodalBlending::render(GPUResourceHandle fbo1, QList<int> nodalTextures, QList<int> nodalWeightTextures)
{
	float width = panoramaWidth;
	float height = panoramaHeight;

	GPUResourceHandle bgColorTextures[8];
	GPUResourceHandle bgWeightTextures[8];
	int bgWeightOn[8];
	for (int i = 0; i < nodalTextures.size(); i++)
		bgColorTextures[i] = nodalTextures[i];
	for (int i = 0; i < nodalWeightTextures.size(); i++)
	{
		bgWeightTextures[i] = nodalWeightTextures[i];
		bgWeightOn[i] = bgWeightTextures[i] != -1;
	}

	runNodalBlending_Kernel(m_cudaTargetSurface, fbo1, bgColorTextures, bgWeightTextures, bgWeightOn, width, height, nodalCameraCount);
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