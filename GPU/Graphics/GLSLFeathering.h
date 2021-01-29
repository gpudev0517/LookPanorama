#ifndef GLSLFEATHERING_H
#define GLSLFEATHERING_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GPUFeathering.h"

/// <summary>
/// The opengl shader that feathers the overlapping area.
/// </summary>
class GLSLFeathering : public GPUFeathering
{
	Q_OBJECT
public:
	explicit GLSLFeathering(QObject *parent = 0);
	virtual ~GLSLFeathering();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount);
	virtual void render(GPUResourceHandle *fbos, GPUResourceHandle *weights, int compositeID = 0);

};

#endif // GLSLFEATHERING_H