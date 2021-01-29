#ifndef CUDAWEIGHTMAP_H
#define CUDAWEIGHTMAP_H

#ifdef USE_CUDA
#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>
#include "Structures.h"

#include "GPUProgram.h"
#include "GPUWeightMap.h"

class CUDACameraWeightMap : public GPUCameraWeightMap
{
	Q_OBJECT
public:
	explicit CUDACameraWeightMap(QObject *parent = 0);
	virtual ~CUDACameraWeightMap();

	virtual void initialize(int imageWidth, int imageHeight);
	virtual void render(int camID);
	virtual void updateCameraParams();
};

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer
class CUDAPanoramaWeightMap : public GPUPanoramaWeightMap
{
	Q_OBJECT
public:
	explicit CUDAPanoramaWeightMap(QObject *parent = 0, bool isYUV = true);
	virtual ~CUDAPanoramaWeightMap();

	virtual void initialize(int xres, int yres, int panoWidth, int panoHeight);
	virtual void render(unsigned int weightTextureId, unsigned int deltaWeightTexId, int camID);
	virtual void updateCameraParams();

	virtual void setCameraInput(CameraInput camIn) {
		camInput = camIn; 
		if (!m_devMat){
			cudaMalloc(&m_devMat, 9 * sizeof(float));
		}
		mat3 m = getCameraViewMatrix(camInput.m_cameraParams.m_yaw, camInput.m_cameraParams.m_pitch, camInput.m_cameraParams.m_roll);
		cudaMemcpy(m_devMat, m.mat_array, 9 * sizeof(float), cudaMemcpyHostToDevice);
	}

	float* m_devMat;
};


class CUDADeltaWeightMap : public GPUDeltaWeightMap
{
	Q_OBJECT
public:
	explicit CUDADeltaWeightMap(QObject *parent = 0, bool isYUV = true);
	virtual ~CUDADeltaWeightMap();

	virtual void initialize(int xres, int yres, int panoWidth, int panoHeight);
	virtual void resetMap();
	virtual GPUResourceHandle getTargetBuffer();
	virtual void refreshCUDAArray();
	virtual void unRegisterCUDAArray();
	virtual void registerCUDAArray();

protected:
	cudaGraphicsResource *m_cudaFboTextureId;
	
};

#endif //USE_CUDA
#endif // CUDAWEIGHTMAP_H