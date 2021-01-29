#ifndef CUDAFINALPANORAMA_H
#define CUDAFINALPANORAMA_H

#ifdef USE_CUDA

#include "GPUProgram.h"
#include <QOpenGLBuffer>
#include "GPUFinalPanorama.h"

/// <summary>
/// The final opengl shader that shows mono and stereo panorama in one view.
/// It can just show one panorama with the resolution of width by height,
/// or can show stereo panorama by top-bottom mode with the resolution of width by height*2.
/// </summary>
class CUDAPanorama : public GPUPanorama
{
	Q_OBJECT
public:
	explicit CUDAPanorama(QObject *parent = 0);
	virtual ~CUDAPanorama();

	virtual void initialize(int panoWidth, int panoHeight, bool isStereo);
	virtual void render(GPUResourceHandle fbos[], bool isOutput);

	virtual void downloadTexture(unsigned char* alphaBuffer);

protected:
	cudaGraphicsResource  *m_cudaFboTextureId;

};

class CUDAFinalPanorama : public GPUFinalPanorama
{
	Q_OBJECT
public:
	explicit CUDAFinalPanorama(QObject *parent = 0);
	virtual ~CUDAFinalPanorama();

	virtual void initialize(int panoWidth, int panoHeight, bool isStereo);
	virtual void render(GPUResourceHandle fbo);

	virtual void downloadTexture(unsigned char* rgbBuffer);

	GPUResourceHandle getTargetGPUResource() { return m_cudaTargetTextures[1 - m_workingGPUResource]; }

private:
	virtual void reconfig(FinalPanoramaConfig config);

	cudaArray *m_cudaTargetArrays[2];
	GPUResourceHandle m_cudaTargetSurfaces[2];
	GPUResourceHandle m_cudaTargetTextures[2];
};

#endif //USE_CUDA

#endif // GLSLFINALPANORAMA_H