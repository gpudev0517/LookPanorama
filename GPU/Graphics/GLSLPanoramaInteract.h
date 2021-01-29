#ifndef GLSL_PANORAMA_INTERACT_H
#define GLSL_PANORAMA_INTERACT_H

#include "GPUProgram.h"
#include "3DMath.h"

/// <summary>
/// The opengl shader that blends all color views with blend map
/// </summary>
class GLSLPanoramaInteract : public GPUProgram
{
	//Q_OBJECT
public:
	explicit GLSLPanoramaInteract(QObject *parent = 0);
	virtual ~GLSLPanoramaInteract();

	void clear();
	void initialize(int panoWidth, int panoHeight);
	void render(GPUResourceHandle originalTextures, bool isStereo, float yaw, float pitch, float roll);

	virtual const int getWidth();
	virtual const int getHeight();

	const mat4& getProjMat(){ return m_projMat; }
	const mat4& getViewMat(){ return m_viewMat; }

private:
	int panoramaWidth;
	int panoramaHeight;

	GLuint m_modelViewMatrix;

	GLfloat *vertices;
	GLfloat *texCoords;
	GLuint m_renderbuffer;

	mat4 m_projMat;
	mat4 m_viewMat;
};

#endif // GLSL_PANORAMA_INTERACT_H