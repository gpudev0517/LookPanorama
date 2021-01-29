#ifndef GLSLBOXBLUR_H
#define GLSLBOXBLUR_H

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that calculates the box blur for each camera view
/// </summary>
class GLSLBoxBlur : public GPUProgram
{
	Q_OBJECT
public:
	explicit GLSLBoxBlur(QObject *parent = 0);
	virtual ~GLSLBoxBlur();

	void initialize(int panoWidth, int panoHeight, bool isVertical, bool isPartial);
	void render(GPUResourceHandle textureId, int blurRadius, float alphaType = 0);

	virtual const int getWidth();
	virtual const int getHeight();

private:
	int panoramaWidth;
	int panoramaHeight;

	bool isPartial;

	GLuint blurRadiusUnif;
	GLuint isPartialUnif;
};

#endif // GLSLBOXBLUR_H