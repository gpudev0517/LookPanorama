#ifndef GLSLLAPLACIAN_H
#define GLSLLAPLACIAN_H

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that calculates the laplacian for each camera view
/// </summary>
class GLSLLaplacian : public GPUProgram
{
	Q_OBJECT
public:
	explicit GLSLLaplacian(QObject *parent = 0);
	virtual ~GLSLLaplacian();

	void initialize(int panoWidth, int panoHeight);
	void render(GPUResourceHandle srcTextureId, GPUResourceHandle gaussianTextureId);

	virtual const int getWidth();
	virtual const int getHeight();

private:
	int panoramaWidth;
	int panoramaHeight;
};

#endif // GLSLLAPLACIAN_H