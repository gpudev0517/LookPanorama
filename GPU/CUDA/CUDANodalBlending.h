#ifndef CUDANODALBLENDING_H
#define CUDANODALBLENDING_H

#ifdef USE_CUDA

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GPUProgram.h"
#include "GPUNodalBlending.h"

/// <summary>
/// The opengl shader that blends two texture with alpha blending (One is nodal background, and one is the foreground)
/// </summary>
class CUDANodalBlending : public GPUNodalBlending
{
	Q_OBJECT
public:
	explicit CUDANodalBlending(QObject *parent = 0);
	virtual ~CUDANodalBlending();

	virtual void initialize(int panoWidth, int panoHeight, int nodalCount, bool haveNodalMaskImage);
	virtual void render(GPUResourceHandle fbo1, QList<int> nodalTextures, QList<int> nodalWeightTextures);
	
};

#endif //USE_CUDA

#endif // CUDANODALBLENDING_H