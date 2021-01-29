#ifndef CUDAUNWARP_H
#define CUDAUNWARP_H

#ifdef USE_CUDA

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>

#include "GPUProgram.h"

#include "D360Parser.h"
#include "common.h"
#include "GPUUnwarp.h"

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer
class CUDAUnwarp : public GPUUnwarp
{
    Q_OBJECT
public:
	explicit CUDAUnwarp(QObject *parent = 0, bool isYUV = true);
	virtual ~CUDAUnwarp();

    virtual void initialize(int id, int xres, int yres, int panoWidth, int panoHeight);
	virtual void render(GPUResourceHandle rgbTextureId, RenderMode renderMode);
	virtual void updateCameraParams();

	virtual void setCameraInput(CameraInput camIn) {
		camInput = camIn; 
		mat3 m = getCameraViewMatrix(camInput.m_cameraParams.m_yaw, camInput.m_cameraParams.m_pitch, camInput.m_cameraParams.m_roll);
		cudaMemcpy(m_devMat, m.mat_array, 9 * sizeof(float), cudaMemcpyHostToDevice);

	}

	float *m_devMat;

};

#endif //USE_CUDA

#endif // CUDAUNWARP_H
