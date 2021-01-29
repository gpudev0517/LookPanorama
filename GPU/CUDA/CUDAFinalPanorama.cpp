
#ifdef USE_CUDA

#include "CUDAFinalPanorama.h"
#include "define.h"
#include "common.h"


extern "C" void runPanorama_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex1, cudaTextureObject_t inputTex2, int width, int height,
	bool isStereo, bool isOutput);
extern "C" void runFinalPanorama_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height);
extern cudaStream_t g_CurStream;
extern cudaStream_t g_NextStream;


CUDAPanorama::CUDAPanorama(QObject *parent) : GPUPanorama(parent)
{
}

CUDAPanorama::~CUDAPanorama()
{
	if (m_initialized)
	{
		cudaGraphicsUnregisterResource(m_cudaFboTextureId);
	}

	
	m_cudaTargetArray = NULL;
}

void CUDAPanorama::initialize(int panoWidth, int panoHeight, bool isStereo)
{
	this->m_stereo = isStereo;
	this->panoramaViewWidth = panoWidth;
	if (m_stereo)
	{
		this->panoramaViewHeight = panoHeight * 2;
	}
	else
	{
		this->panoramaViewHeight = panoHeight;
	}

	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panoramaViewWidth, panoramaViewHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	cudaGraphicsGLRegisterImage(&m_cudaFboTextureId, m_fboTextureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &m_cudaFboTextureId, 0);
	cudaGraphicsSubResourceGetMappedArray(&m_cudaTargetArray, m_cudaFboTextureId, 0, 0);
	cudaGraphicsUnmapResources(1, &m_cudaFboTextureId, 0);

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


void CUDAPanorama::render(GPUResourceHandle fbos[], bool isOutput)
{
	float width = panoramaViewWidth;
	float height = panoramaViewHeight;

	runPanorama_Kernel(m_cudaTargetSurface, fbos[0], fbos[1], width, height, m_stereo, isOutput);
#if 0
	GLubyte *buffer = new GLubyte[panoramaViewWidth * panoramaViewHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaViewWidth *panoramaViewHeight * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaViewWidth, panoramaViewHeight, QImage::Format_RGBA8888);
	img.save(QString("Panorama_") + ".png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
	
}

void CUDAPanorama::downloadTexture(unsigned char* alphaBuffer)
{
	if (m_initialized){
		uchar *rgbaBuffer = new uchar[getWidth() * getHeight() * 4];
		cudaMemcpyFromArray(rgbaBuffer, m_cudaTargetArray, 0, 0, getWidth() * getHeight() * 4, cudaMemcpyDeviceToHost);
		for (int j = 0; j < getHeight(); j++){
			for (int i = 0; i < getWidth(); i++){
				alphaBuffer[j * getWidth() + i] = rgbaBuffer[4 * (j * getWidth() + i) + 3];
			}
		}
		delete[] rgbaBuffer;
	}
}

// Final Panorama
CUDAFinalPanorama::CUDAFinalPanorama(QObject *parent) : GPUFinalPanorama(parent)
{
}

CUDAFinalPanorama::~CUDAFinalPanorama()
{
	if (m_initialized)
	{
		for (int i = 0; i < 2; i++)
		{
			cudaFreeArray(m_cudaTargetArrays[i]);
			cudaDestroySurfaceObject(m_cudaTargetSurfaces[i]);
			cudaDestroyTextureObject(m_cudaTargetTextures[i]);
		}
	}
}

void CUDAFinalPanorama::initialize(int panoWidth, int panoHeight, bool isStereo)
{
	reconfig(FinalPanoramaConfig(panoWidth, panoHeight, isStereo));
}

void CUDAFinalPanorama::reconfig(FinalPanoramaConfig config)
{
	if (config.panoWidth == panoramaViewWidth && config.panoHeight == panoramaViewHeight && config.isStereo == m_stereo)
		return;

	if (m_initialized)
	{
		for (int i = 0; i < 2; i++)
		{
			cudaFreeArray(m_cudaTargetArrays[i]);
			cudaDestroySurfaceObject(m_cudaTargetSurfaces[i]);
			cudaDestroyTextureObject(m_cudaTargetTextures[i]);
		}
	}

	this->m_stereo = config.isStereo;
	this->panoramaViewWidth = config.panoWidth;
	if (m_stereo)
	{
		this->panoramaViewHeight = config.panoHeight * 2;
	}
	else
	{
		this->panoramaViewHeight = config.panoHeight;
	}
	this->panoramaViewHeight = this->panoramaViewHeight * 3 / 2;

	for (int i = 0; i < 2; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaTargetArrays[i], &channelFormat, panoramaViewWidth, panoramaViewHeight, cudaArraySurfaceLoadStore);

		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = m_cudaTargetArrays[i];
		cudaCreateSurfaceObject(&m_cudaTargetSurfaces[i], &surfRes);

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;

		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.readMode = cudaReadModeNormalizedFloat;

		cudaCreateTextureObject(&m_cudaTargetTextures[i], &surfRes, &texDescr, NULL);
	}

	m_workingGPUResource = 0;
	m_targetArrayIndex = 0;
	m_initialized = true;
}

void CUDAFinalPanorama::render(GPUResourceHandle fbo)
{
	if (newConfig.size() != 0)
	{
		reconfig(newConfig[0]);
		newConfig.clear();
	}
	float width = panoramaViewWidth;
	float height = panoramaViewHeight;
	runFinalPanorama_Kernel(m_cudaTargetSurfaces[m_workingGPUResource], fbo, width, height);
	//cudaStreamSynchronize(g_CurStream);
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaViewWidth * panoramaViewHeight];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, panoramaViewWidth *panoramaViewHeight, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaViewWidth, panoramaViewHeight, QImage::Format_Grayscale8);
	img.save(QString("FinalPanorama.png"));
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
	m_workingGPUResource = 1 - m_workingGPUResource;

}

void CUDAFinalPanorama::downloadTexture(unsigned char* rgbBuffer)
{
	if (m_initialized) {
		m_targetArrayIndex = (m_targetArrayIndex + 1) % 2;
		cudaStream_t stream = m_targetArrayIndex == 0 ? g_CurStream : g_NextStream;
		cudaMemcpyFromArrayAsync(rgbBuffer, m_cudaTargetArrays[1 - m_workingGPUResource], 0, 0, getWidth() * getHeight(), cudaMemcpyDeviceToHost, stream);
		//cudaStreamSynchronize(stream);
	}
}
#endif //USE_CUDA