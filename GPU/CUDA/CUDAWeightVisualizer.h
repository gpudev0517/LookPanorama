#ifndef CUDAWEIGHTVISUALIZER_H
#define CUDAWEIGHTVISUALIZER_H

#ifdef USE_CUDA

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>

#include "common.h"

#include "GPUProgram.h"
#include "GPUWeightVisualizer.h"

class CUDAWeightVisualizer : public GPUWeightVisualizer
{
	Q_OBJECT
public:
	explicit CUDAWeightVisualizer(QObject *parent = 0);
	virtual ~CUDAWeightVisualizer();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount);
	virtual void render(WeightMapPaintMode paintMode, GPUResourceHandle fbos[], GPUResourceHandle weights[], int compositeID, int camIndex, int eyeMode);

protected:

	GPUResourceHandle *m_devFbos;
	GPUResourceHandle *m_devWeights;

	
};

#endif //USE_CUDA

#endif // CUDAWEIGHTVISUALIZER_H