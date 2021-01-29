#ifndef GLSLPANO_POSTPROCESSING_H
#define GLSLPANO_POSTPROCESSING_H

#include "GPUPanoramaPostProcessing.h"

class GLSLPanoramaPostProcessing : public GPUPanoramaPostProcessing
{
	Q_OBJECT
public:
	explicit GLSLPanoramaPostProcessing(QObject *parent = 0);
	virtual ~GLSLPanoramaPostProcessing();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void render(GLuint panoTextureId, vec3 ctLightColor,
		float yaw, float pitch, float roll,
		int seamTextureId);

	virtual	void setLutData(QVariantList *vList);

	void updateGlobalParams(float yaw, float pitch, float roll)
	{

	}

private:


	GLuint m_lutTexture;
	QOpenGLBuffer *m_lutPBO;
	GLuint placeMatUnif;
};

#endif // GLSLPANO_COLORCORRECTION_H