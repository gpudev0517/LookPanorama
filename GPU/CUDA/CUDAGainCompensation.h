#ifndef CUDAGAINCOMPENSATION_H
#define CUDAGAINCOMPENSATION_H

#ifdef USE_CUDA

#include "GPUGainCompensation.h"
#include <QOpenGLBuffer>

/// <summary>
/// The opengl shader that compensates the gain.
/// </summary>
class CUDAGainCompensation : public GPUGainCompensation
{
	Q_OBJECT
public:
	explicit CUDAGainCompensation(QObject *parent = 0);
	virtual ~CUDAGainCompensation();

	void initialize(int cameraWidth, int cameraHeight);
	void render(GPUResourceHandle textureId, float ev);
	
	void getRGBBuffer(unsigned char* rgbBuffer);

	GPUResourceHandle getTargetGPUResource() { return m_cudaTargetTextures[getReadyGPUResourceIndex()]; }
	GPUResourceHandle getRawGPUResource() { return m_fboTextureIds[getReadyGPUResourceIndex()]; }

private:

	GLuint m_fboTextureIds[m_targetCount];
	cudaGraphicsResource *m_cudaFboTextureId[m_targetCount];

	cudaSurfaceObject_t m_cudaTargetSurfaces[m_targetCount];
	cudaArray *m_cudaTargetArrays[m_targetCount];
	cudaSurfaceObject_t m_cudaTargetTextures[m_targetCount];
};

#endif //USE_CUDA

#endif // CUDAGAINCOMPENSATION_H