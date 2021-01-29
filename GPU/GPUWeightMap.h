#ifndef GPUWEIGHTMAP_H
#define GPUWEIGHTMAP_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>
#include "Structures.h"

#include "GPUProgram.h"

class GPUCameraWeightMap : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUCameraWeightMap(QObject *parent = 0);
	virtual ~GPUCameraWeightMap();

	virtual void initialize(int imageWidth, int imageHeight) = 0;
	virtual void render(int camID) = 0;
	virtual void updateCameraParams() = 0;;

	void setCameraInput(CameraInput camIn) { camInput = camIn; }

protected:
	int imageWidth;
	int imageHeight;

	CameraInput camInput;
};

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer
class GPUPanoramaWeightMap : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUPanoramaWeightMap(QObject *parent = 0, bool isYUV = true);
	virtual ~GPUPanoramaWeightMap();

	virtual void initialize(int xres, int yres, int panoWidth, int panoHeight) = 0;
	virtual void render(unsigned int weightTextureId, unsigned int deltaWeightTexId, int camID) = 0;
	virtual void updateCameraParams() = 0;

	virtual void setCameraInput(CameraInput camIn) { camInput = camIn; }

	int inputWidth;
	int inputHeight;

	int panoramaWidth;
	int panoramaHeight;

protected:
	CameraInput camInput;
};

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer



class GPUDeltaWeightMap : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUDeltaWeightMap(QObject *parent = 0, bool isYUV = true);
	virtual ~GPUDeltaWeightMap();

	virtual void initialize(int xres, int yres, int panoWidth, int panoHeight);
	void saveWeightmap(QString filename);
	bool loadWeightmap(QString filename);
	void render(float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM, int camID);
	virtual void resetMap();

	virtual void refreshCUDAArray() = 0;
	virtual void unRegisterCUDAArray() = 0;
	virtual void registerCUDAArray() = 0;
	
	GPUResourceHandle getTargetGPUResourceForUndoRedo(){ return m_fboTextureId; }

	void setCameraInput(CameraInput camIn) { camInput = camIn; }
	void updateCameraParams();

	int camWidth;
	int camHeight;

	int panoramaWidth;
	int panoramaHeight;

	static GLfloat *vertices;

	QString m_Name;

protected:
	CameraInput camInput;
	CameraData cam;
	GLuint ubo;

};
#endif // GPUWEIGHTMAP_H