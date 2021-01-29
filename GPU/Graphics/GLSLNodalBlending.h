#ifndef GLSLNODALBLENDING_H
#define GLSLNODALBLENDING_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GPUNodalBlending.h"

/// <summary>
/// The opengl shader that blends two texture with alpha blending (One is nodal background, and one is the foreground)
/// </summary>
class GLSLNodalBlending : public GPUNodalBlending
{
	Q_OBJECT
public:
	explicit GLSLNodalBlending(QObject *parent = 0);
	virtual ~GLSLNodalBlending();

	virtual void initialize(int panoWidth, int panoHeight, int nodalCameraCount, bool haveNodalMaskImage);
	virtual void render(GPUResourceHandle fbo1, QList<int> nodalTextures, QList<int> nodalWeightTextures);
};

#endif // GLSLNODALBLENDING_H