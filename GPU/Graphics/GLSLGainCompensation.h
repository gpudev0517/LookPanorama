#ifndef GLSLGAINCOMPENSATION_H
#define GLSLGAINCOMPENSATION_H

#include "GPUProgram.h"
#include "GPUGainCompensation.h"
#include <QOpenGLBuffer>

/// <summary>
/// The opengl shader that compensates the gain.
/// </summary>
class GLSLGainCompensation : public GPUGainCompensation
{
	Q_OBJECT
public:
	explicit GLSLGainCompensation(QObject *parent = 0);
	virtual ~GLSLGainCompensation();

	void initialize(int cameraWidth, int cameraHeight);
	void render(GPUResourceHandle textureId, float ev);
	
	void getRGBBuffer(unsigned char* rgbBuffer);

	
	GPUResourceHandle getTargetGPUResource() { return m_fboTextureIds[getReadyGPUResourceIndex()]; }
	GPUResourceHandle getRawGPUResource() { return m_fboTextureIds[getReadyGPUResourceIndex()]; }

private:
	GLuint m_gainUnif;

	GLuint m_fboIds[m_targetCount];
	GLuint m_fboTextureIds[m_targetCount];

	QOpenGLBuffer* m_pbos[2];
};

#endif // GLSLGAINCOMPENSATION_H