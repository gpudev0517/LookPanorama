#pragma once
#include <QtCore/QThread>

#include <QOpenGLFunctions>
#include <QtGui/QOpenGLShaderProgram>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFunctions_2_0>

#include "GLSLColorCvt.h"
#include "GLSLGainCompensation.h"
#include "GLSLUnwarp.h"
#include "GLSLWeightMap.h"

#ifdef USE_CUDA
#include "CUDAColorCvt.h"
#include "CUDAGainCompensation.h"
#include "CUDAUnwarp.h"
#include "CUDAWeightMap.h"
#endif

class ICamera
{
public:
	virtual GPUResourceHandle getUnwarpTexture() = 0;
	virtual GPUResourceHandle getTargetFrameBuffer() = 0;
	virtual GPUResourceHandle getRawTexture() = 0;
	virtual GPUResourceHandle getWeightTexture() = 0;

	virtual void updateCameraParams() = 0;
};

/// <summary>
// Single camera image processing pipeline.
/// </summary>
class SingleViewProcess : public QObject, public ICamera
{
	Q_OBJECT

public:
	SingleViewProcess(SharedImageBuffer *sharedImageBuffer, int deviceIndex, int width, int height);
	virtual ~SingleViewProcess();

	void	init();
	void	clear();

	void create(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3);
	void destroy();

	void uploadTexture(ImageBufferData mat);
	void render();
	void downloadTexture(unsigned char* liveGrabBuffer);

	/*void	stop();
	bool isFinished();
	void waitForFinish();*/

	virtual GPUResourceHandle getUnwarpTexture();
	virtual GPUResourceHandle getTargetFrameBuffer();
	virtual GPUResourceHandle getRawTexture();
	virtual GPUResourceHandle getWeightTexture();
	virtual GPUResourceHandle getColorCvtTexture();
	virtual GPUResourceHandle getColorCvtBuffer(){
		return m_2rgbColorCvt->getTargetBuffer();
	}
	virtual void updateCameraParams();

	/*void processImgProc(ImageBufferData mat);
	void started();*/

private:
	SharedImageBuffer *sharedImageBuffer;

	int camIndex;
	int width;
	int height;
	
	void processIndividualViews(bool weightMapChangedByCameraParam);
	int getCamIndex();

	/*QOffscreenSurface* m_surface;
	QOpenGLContext* m_context;
	QOpenGLFunctions_2_0* functions_2_0;*/

	GPUColorCvt_2RGBA* m_2rgbColorCvt;
	GPUGainCompensation* m_gainCompensation;
	GPUUnwarp* m_unwarp;
	GPUCameraWeightMap* m_camWeightMap;

	ImageBufferData m_frame;

	bool m_paramChanged;

	/*// thread
	QMutex doStopMutex;
	QMutex doPauseMutex;

	QMutex finishMutex;
	QWaitCondition finishWC;
	bool m_finished;

	bool doStop;

	QMutex frameMutex;
	bool isFrameReady;

protected:
	void run();

public slots:
	void process();*/
};
