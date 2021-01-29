#ifndef CUDAFEATHERING_H
#define CUDAFEATHERING_H

#ifdef USE_CUDA
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GPUProgram.h"
#include "GPUFeathering.h"

/// <summary>
/// The opengl shader that feathers the overlapping area.
/// </summary>
class CUDAFeathering : public GPUFeathering
{
	Q_OBJECT
public:
	explicit CUDAFeathering(QObject *parent = 0);
	virtual ~CUDAFeathering();

	void initialize(int panoWidth, int panoHeight, int viewCount);
	void render(GPUResourceHandle *fbos, GPUResourceHandle *weights, int compositeID = 0);

private:


};

#endif //USE_CUDA

#endif // CUDAFEATHERING_H