#ifndef GLSLFINALPANORAMA_H
#define GLSLFINALPANORAMA_H

#include "GPUProgram.h"
#include <QOpenGLBuffer>
#include "GPUFinalPanorama.h"

/// <summary>
/// The final opengl shader that shows mono and stereo panorama in one view.
/// It can just show one panorama with the resolution of width by height,
/// or can show stereo panorama by top-bottom mode with the resolution of width by height*2.
/// </summary>
class GLSLPanorama : public GPUPanorama
{
	Q_OBJECT
public:
	explicit GLSLPanorama(QObject *parent = 0);
	virtual ~GLSLPanorama();

	virtual void initialize(int panoWidth, int panoHeight, bool isStereo);
	virtual void render(GPUResourceHandle fbos[], bool isOutput);

	virtual void downloadTexture(unsigned char* alphaBuffer);
};

class GLSLFinalPanorama : public GPUFinalPanorama
{
	Q_OBJECT
public:
	explicit GLSLFinalPanorama(QObject *parent = 0);
	virtual ~GLSLFinalPanorama();

	virtual void initialize(int panoWidth, int panoHeight, bool isStereo);
	virtual void render(GPUResourceHandle fbo);

	virtual void downloadTexture(unsigned char* rgbBuffer);

	virtual GPUResourceHandle getTargetGPUResource() { return m_fboTextureIds[1 - m_workingGPUResource]; }

protected:
	void reconfig(FinalPanoramaConfig config);
	GLuint m_fboIds[2];
	GLuint m_fboTextureIds[2];

	// gl functions
	QOpenGLBuffer* m_pbos[2];
};

#endif // GLSLFINALPANORAMA_H