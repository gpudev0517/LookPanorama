#ifndef GLSLBLENDING_H
#define GLSLBLENDING_H

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that blends all color views with blend map
/// </summary>
class GLSLBlending : public GPUProgram
{
	Q_OBJECT
public:
	explicit GLSLBlending(QObject *parent = 0);
	virtual ~GLSLBlending();

	void initialize(int panoWidth, int panoHeight, int viewCount);
	void render(GPUResourceHandle colorTextures[], GPUResourceHandle alphaTextures[], int colorLevel, int alphaLevel, int alphaType = 0);

	virtual const int getWidth();
	virtual const int getHeight();

private:
	float getLevelScale(int level);

	int panoramaWidth;
	int panoramaHeight;

	int m_viewCount;

	GLuint levelScaleUnif;
};

#endif // GLSLBLENDING_H