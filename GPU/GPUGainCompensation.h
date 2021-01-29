#ifndef GPUGAINCOMPENSATION_H
#define GPUGAINCOMPENSATION_H

#include "GPUProgram.h"
#include <QOpenGLBuffer>

/// <summary>
/// The opengl shader that compensates the gain.
/// </summary>
class GPUGainCompensation : public GPUProgram
{
	Q_OBJECT
public:
	GPUGainCompensation(QObject *parent);
	virtual ~GPUGainCompensation();

	virtual void initialize(int cameraWidth, int cameraHeight) = 0;
	virtual void render(GPUResourceHandle textureId, float ev) = 0;
	
	virtual void getRGBBuffer(unsigned char* rgbBuffer) = 0;

	virtual GPUResourceHandle getRawGPUResource() = 0;

	virtual const int getWidth();
	virtual const int getHeight();


protected:

	int cameraWidth;
	int cameraHeight;

	static const int m_targetCount = 2;
	int m_workingGPUResource;
	int m_targetArrayIndex;

	int getReadyGPUResourceIndex();
};

#endif // GPUGAINCOMPENSATION_H