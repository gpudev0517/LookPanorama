#ifndef GLSLPANORAMAWEIGHT_H
#define GLSLPANORAMAWEIGHT_H

#include "GLSLWeightMap.h"

/// <summary>
/// The opengl shader that has delta weight map
/// </summary>
class GLSLPanoramaWeight : public QObject
{
	Q_OBJECT
public:
	explicit GLSLPanoramaWeight(QObject *parent = 0);
	virtual ~GLSLPanoramaWeight();

	void setGL(QOpenGLFunctions* gl);
	void initialize(int camWidth, int camHeight, int panoWidth, int panoHeight);
	
	void renderWeight(GLuint srcWeight, int camID);
	void renderDelta(float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM, int camID);
	void resetDeltaWeight();

	int getPanoWeightTexture() { return m_panoWeightMap->getTargetGPUResource(); }
	int getDeltaWeightTexture() { return m_deltaWeightMap->getTargetGPUResource(); }
	int getDeltaWeightFrameBuffer() { return m_deltaWeightMap->getTargetFrameBuffer(); }

	void setCameraInput(CameraInput camInput);
	void updateCameraParams();

	void saveWeightmap(QString filename);
	void loadWeightmap(QString filename);

private:
	int camWidth;
	int camHeight;
	int panoramaWidth;
	int panoramaHeight;

	// gl functions
	QOpenGLFunctions* m_gl;

	bool m_initialized;

	GLSLPanoramaWeightMap* m_panoWeightMap;
	GLSLDeltaWeightMap* m_deltaWeightMap;
};

#endif // GLSLPANORAMAWEIGHT_H