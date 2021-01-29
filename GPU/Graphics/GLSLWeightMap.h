#ifndef GLSLWEIGHTMAP_H
#define GLSLWEIGHTMAP_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>
#include "Structures.h"

#include "GPUProgram.h"
#include "GPUWeightMap.h"

class GLSLCameraWeightMap : public GPUCameraWeightMap
{
	Q_OBJECT
public:
	explicit GLSLCameraWeightMap(QObject *parent = 0);
	virtual ~GLSLCameraWeightMap();

	virtual void initialize(int imageWidth, int imageHeight);
	virtual void render(int camID);
	virtual void updateCameraParams();

private:

	// undistort GLSL program
	int m_fisheyeLensRadiusRatio1Unif;
	int m_fisheyeLensRadiusRatio2Unif;
	int m_blendCurveStartUnif;

	GLuint xrad1Unif;
	GLuint xrad2Unif;
	GLuint yrad1Unif;
	GLuint yrad2Unif;
};

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer
class GLSLPanoramaWeightMap : public GPUPanoramaWeightMap
{
	Q_OBJECT
public:
	explicit GLSLPanoramaWeightMap(QObject *parent = 0, bool isYUV = true);
	virtual ~GLSLPanoramaWeightMap();

	virtual void initialize(int xres, int yres, int panoWidth, int panoHeight);
	virtual void render(unsigned int weightTextureId, unsigned int deltaWeightTexId, int camID);
	virtual void updateCameraParams();

private:
	CameraData cam;
	GLuint ubo;
};

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer



class GLSLDeltaWeightMap : public GPUDeltaWeightMap
{
public:

	virtual void refreshCUDAArray()
	{

	}
	virtual void unRegisterCUDAArray()
	{

	}

	virtual void registerCUDAArray()
	{

	}
};
#endif // GLSLWEIGHTMAP_H