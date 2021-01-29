#ifndef GPUNODALBLENDING_H
#define GPUNODALBLENDING_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that blends two texture with alpha blending (One is nodal background, and one is the foreground)
/// </summary>
class GPUNodalBlending : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUNodalBlending(QObject *parent = 0);
	virtual ~GPUNodalBlending();

	virtual void initialize(int panoWidth, int panoHeight, int nodalCount, bool haveNodalMaskImage) = 0;
	virtual void render(GPUResourceHandle fbo1, QList<int> nodalTextures, QList<int> nodalWeightTextures) = 0;

protected:
	int panoramaWidth;
	int panoramaHeight;
	int nodalCameraCount;
	bool haveNodalMaskImage;
};

#endif // GPUNODALBLENDING_H