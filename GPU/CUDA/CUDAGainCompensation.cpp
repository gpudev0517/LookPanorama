
#ifdef USE_CUDA

#include "CUDAGainCompensation.h"
#include "common.h"
#include "define.h"

extern "C" void runGainCompensation_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, float gain);
extern cudaStream_t g_CurStream;
extern cudaStream_t g_NextStream;

CUDAGainCompensation::CUDAGainCompensation(QObject *parent) : GPUGainCompensation(parent)
{
	m_fboTextureIds[0] = m_fboTextureIds[1] = -1;
	m_cudaTargetArrays[0] = m_cudaTargetArrays[0] = NULL;
	m_cudaTargetSurfaces[0] = m_cudaTargetSurfaces[1] = -1;
}

CUDAGainCompensation::~CUDAGainCompensation()
{
	if (m_initialized)
	{

		for (int i = 0; i < m_targetCount; i++){
			if (m_cudaTargetArrays[0]){
				cudaGraphicsUnregisterResource(m_cudaFboTextureId[i]);
//				cudaFreeArray(m_cudaTargetArrays[i]);
				cudaDestroySurfaceObject(m_cudaTargetSurfaces[i]);
			}

		}
		if (m_fboTextureIds[0] != -1){
			m_gl->glDeleteTextures(m_targetCount, m_fboTextureIds);
		}
			
		m_fboTextureIds[0] = m_fboTextureIds[1] = -1;
		m_cudaTargetArrays[0] = m_cudaTargetArrays[0] = NULL;
		m_cudaTargetSurfaces[0] = m_cudaTargetSurfaces[1] = -1;
	}
}

void CUDAGainCompensation::initialize(int cameraWidth, int cameraHeight)
{
	this->cameraWidth = cameraWidth;
	this->cameraHeight = cameraHeight;
	
	// frame buffer
	m_gl->glGenTextures(m_targetCount, m_fboTextureIds);
	
	for (int i = 0; i < m_targetCount; i++)
	{
		m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureIds[i]);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, getWidth(), getHeight(), 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

		
		cudaGraphicsGLRegisterImage(&m_cudaFboTextureId[i], m_fboTextureIds[i], GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
		cudaGraphicsMapResources(1, &m_cudaFboTextureId[i], 0);
		cudaGraphicsSubResourceGetMappedArray(&m_cudaTargetArrays[i], m_cudaFboTextureId[i], 0, 0);
		cudaGraphicsUnmapResources(1, &m_cudaFboTextureId[i], 0);
 
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


void CUDAGainCompensation::render(GPUResourceHandle textureId, float ev)
{
	float gain = ev2gain(ev);
	runGainCompensation_Kernel(m_cudaTargetSurfaces[m_workingGPUResource], textureId, cameraWidth, cameraHeight, gain);
#if 0
	GLubyte *buffer = new GLubyte[cameraWidth * cameraHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, cameraWidth *cameraHeight * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, cameraWidth, cameraHeight, QImage::Format_RGBA8888);
	img.save("GainCompensation.png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
	m_workingGPUResource = (m_workingGPUResource + 1) % m_targetCount;
}

void CUDAGainCompensation::getRGBBuffer(unsigned char* rgbBuffer)
{
	if (rgbBuffer == NULL)
	{
		return;
	}
	if (m_initialized) {
		m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
		cudaStream_t stream = m_targetArrayIndex == 0 ? g_CurStream : g_NextStream;
		cudaMemcpyFromArrayAsync(rgbBuffer, m_cudaTargetArrays[getReadyGPUResourceIndex()], 0, 0, getWidth() * getHeight() * 3, cudaMemcpyDeviceToHost, stream);
		//cudaStreamSynchronize(stream);
	}
}


#endif //USE_CUDA