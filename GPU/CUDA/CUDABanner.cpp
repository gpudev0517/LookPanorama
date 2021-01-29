
#ifdef USE_CUDA

#include "CUDABanner.h"
#include "define.h"

extern "C" void runBillClear_Kernel(cudaSurfaceObject_t outputSurf, int width, int height);
extern "C" void runBill_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t bgInputTex, cudaTextureObject_t bannerInputTex, int width, int height,
	float *bannerPaiPlane, float bannerPaiZdotOrg, float *bannerHomography, int widthPixels, int heightPixels, int imgWidth, int imgHeight);
extern "C" void runBanner_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t bgInputTex, cudaTextureObject_t bannerInputTex, int width, int height);

// CUDABill
CUDABill::CUDABill(QObject *parent) : GPUBill(parent)
{
	m_devPaiPlane = NULL;
	m_devHomography = NULL;
}

CUDABill::~CUDABill()
{
	cudaFree(m_devPaiPlane);
	cudaFree(m_devHomography);
}

void CUDABill::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

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

void CUDABill::render(std::vector<BannerInfo*> bannerInputs)
{
// 	uchar *src = new uchar[panoramaWidth * panoramaHeight * 4];
// 	memset(src, 0, panoramaWidth * panoramaHeight * 4);
// 	cudaMemcpyToArray(m_cudaTargetArray, 0, 0, src, panoramaWidth * panoramaHeight * 4, cudaMemcpyHostToDevice);
//	delete[] src;
	runBillClear_Kernel(m_cudaTargetSurface, panoramaWidth, panoramaHeight);
	for (int i = 0; i < bannerInputs.size(); i++)
	{
		BannerInfo* pBanner = bannerInputs[i];
		if (!m_devPaiPlane){
			cudaMalloc(&m_devPaiPlane, 16 * sizeof(float));
			cudaMemcpy(m_devPaiPlane, pBanner->paiPlane.mat_array, 16 * sizeof(float), cudaMemcpyHostToDevice);
			
		}
		if (!m_devHomography){
			cudaMalloc(&m_devHomography, 9 * sizeof(float));
			cudaMemcpy(m_devHomography, pBanner->homography.mat_array, 9 * sizeof(float), cudaMemcpyHostToDevice);
		}
		runBill_Kernel(m_cudaTargetSurface, m_cudaTargetTexture, pBanner->billColorCvt->getTargetGPUResource(), panoramaWidth, panoramaHeight, m_devPaiPlane, pBanner->paiZdotOrg,
			m_devHomography, panoramaWidth, panoramaHeight, pBanner->billColorCvt->ImageWidth(), pBanner->billColorCvt->ImageHeight());
		
#if 0
		cudaDeviceSynchronize();
		cudaError err = cudaGetLastError();
		GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight * 4];
		err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight * 4, cudaMemcpyDeviceToHost);
		QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_RGBA8888);
		img.save(QString("Bill") + ".png");
		delete[] buffer;
		if (err != cudaSuccess)
		{
			int a = 0;
			a++;
		}
#endif 
	}
}

// GLSLBanner
CUDABanner::CUDABanner(QObject *parent) : GPUBanner(parent)
{
}

CUDABanner::~CUDABanner()
{
}

void CUDABanner::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;
	
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

void CUDABanner::render(GPUResourceHandle srcTextureId, GPUResourceHandle billTextureId)
{
	runBanner_Kernel(m_cudaTargetSurface, srcTextureId, billTextureId, panoramaWidth, panoramaHeight);

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_RGBA8888);
	img.save(QString("Banner") + ".png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif

}

#endif //USE_CUDA