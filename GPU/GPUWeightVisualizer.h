#ifndef GPUWEIGHTVISUALIZER_H
#define GPUWEIGHTVISUALIZER_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>

#include "common.h"

#include "GPUProgram.h"

class GPUWeightVisualizer : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUWeightVisualizer(QObject *parent = 0);
	virtual ~GPUWeightVisualizer();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount) = 0;
	virtual void render(WeightMapPaintMode paintMode, GPUResourceHandle fbos[], GPUResourceHandle weights[], int compositeID, int camIndex, int eyeMode) = 0;

protected:
	int panoramaWidth;
	int panoramaHeight;

	int m_viewCount;
};

#endif // GPUWEIGHTVISUALIZER_H