#ifndef GLSLRESIZE_H
#define GLSLRESIZE_H

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that resize the texture in half
/// </summary>
class GLSLResize : public GPUProgram
{
	Q_OBJECT
public:
	explicit GLSLResize(QObject *parent = 0);
	virtual ~GLSLResize();

	void initialize(int panoWidth, int panoHeight);
	void render(GPUResourceHandle textureId);

	virtual const int getWidth();
	virtual const int getHeight();

private:
	int panoramaWidth;
	int panoramaHeight;
};

#endif // GLSLRESIZE_H