#ifndef GLSLGAUSSIANBLUR_H
#define GLSLGAUSSIANBLUR_H

#include "GPUProgram.h"
#include "GLSLBoxBlur.h"

/// <summary>
/// The opengl shader that calculates the gaussian blur for each camera view
/// </summary>
class GLSLGaussianBlur : public GPUProgram
{
	Q_OBJECT
public:
	explicit GLSLGaussianBlur(QObject *parent = 0);
	virtual ~GLSLGaussianBlur();

	void initialize(int panoWidth, int panoHeight, bool isPartial);
	void render(GPUResourceHandle textureId, float alphaType = 0);

	virtual GPUResourceHandle getTargetGPUResource();

	virtual const int getWidth();
	virtual const int getHeight();

	GPUResourceHandle boxBlur(GPUResourceHandle src, float radius, float alphaType = 0);
	static void getGaussianRadius();

private:
	int panoramaWidth;
	int panoramaHeight;

	GLSLBoxBlur * m_horizontalBoxBlur;
	GLSLBoxBlur * m_verticalBoxBlur;

	static bool isGaussianRadiusInitialized;
	static int gaussianRadius[3];
};

#endif // GLSLGAUSSIANBLUR_H