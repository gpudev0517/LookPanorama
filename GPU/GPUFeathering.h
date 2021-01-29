#ifndef GPUFEATHERING_H
#define GPUFEATHERING_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "GPUProgram.h"

/// <summary>
/// The opengl shader that feathers the overlapping area.
/// </summary>
class GPUFeathering : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUFeathering(QObject *parent = 0);
	virtual ~GPUFeathering();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount) = 0;
	virtual void render(GPUResourceHandle *fbos, GPUResourceHandle *weights, int compositeID = 0) = 0;

protected:
	int panoramaWidth;
	int panoramaHeight;

	int m_viewCount;
};

#endif // GPUFEATHERING_H