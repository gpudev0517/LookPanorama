#ifndef GLSLWEIGHTVISUALIZER_H
#define GLSLWEIGHTVISUALIZER_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>

#include "common.h"

#include "GPUWeightVisualizer.h"

class GLSLWeightVisualizer : public GPUWeightVisualizer
{
	Q_OBJECT
public:
	explicit GLSLWeightVisualizer(QObject *parent = 0);
	virtual ~GLSLWeightVisualizer();

	virtual void initialize(int panoWidth, int panoHeight, int viewCount);
	virtual void render(WeightMapPaintMode paintMode, GPUResourceHandle fbos[], GPUResourceHandle weights[], int compositeID, int camIndex, int eyeMode);

};

#endif // GLSLWEIGHTVISUALIZER_H