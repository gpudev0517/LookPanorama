#ifndef GPUUNWARP_H
#define GPUUNWARP_H

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QOpenGLTexture>
#include <QOpenGLBuffer>

#include "GPUProgram.h"

#include "D360Parser.h"
#include "common.h"

struct FisheyeCamera
{
    float m_cameraNumber;
    float m_imageWidth;
    float m_imageHeight;
    float m_focal;
    float m_radius;
    float m_fisheyeRadius;
    float m_cx;
    float m_cy;
    float m_offsetx;
    float m_offsety;

    float m_k1;
    float m_k2;
    float m_k3;
    float m_vx;
    float m_vy;
    float m_vz;
    float m_maxcos;

    float m_pitch;
    float m_roll;
    float m_yaw;
};

// undistorts fisheye-lens camera image to panorama image coordinate and render it to FrameBuffer
class GPUUnwarp : public GPUProgram
{
    Q_OBJECT
public:
	explicit GPUUnwarp(QObject *parent = 0, bool isYUV = true);
	virtual ~GPUUnwarp();

	enum RenderMode
	{
		Color,
		WeightMap
	};

	virtual void initialize(int id, int xres, int yres, int panoWidth, int panoHeight) = 0;
	virtual void render(GPUResourceHandle rgbTextureId, RenderMode renderMode) = 0;
	virtual void updateCameraParams() = 0;

	virtual void setCameraInput(CameraInput camIn) { camInput = camIn; }

	int inputWidth;
	int inputHeight;

    int panoramaWidth;
    int panoramaHeight;

	int camID;
protected:
	CameraInput camInput;
};

#endif // GPUUNWARP_H
