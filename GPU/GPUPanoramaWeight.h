#ifndef GPUPANORAMAWEIGHT_H
#define GPUPANORAMAWEIGHT_H

#include "GLSLWeightMap.h"

#ifdef USE_CUDA
#include "CUDAWeightMap.h"
#endif

/// <summary>
/// The opengl shader that has delta weight map
/// </summary>
class GPUPanoramaWeight : public QObject
{
	Q_OBJECT
public:
	explicit GPUPanoramaWeight(QObject *parent = 0);
	virtual ~GPUPanoramaWeight();

	void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3);
	void initialize(int camWidth, int camHeight, int panoWidth, int panoHeight);
	
	void renderWeight(GPUResourceHandle srcWeight, int camID);
	void renderDelta(float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM, int camID);
	void resetDeltaWeight();

	int getPanoWeightTexture() { return m_panoWeightMap->getTargetGPUResource(); }
	int getDeltaWeightTexture() { return m_deltaWeightMap->getTargetGPUResource(); }
	int getDeltaWeightTextureForUndoRedo() { return m_deltaWeightMap->getTargetGPUResourceForUndoRedo(); }
	int getDeltaWeightFrameBuffer() { return m_deltaWeightMap->getTargetBuffer(); }

	void setCameraInput(CameraInput camInput);
	void updateCameraParams();

	void saveWeightmap(QString filename);
	void loadWeightmap(QString filename);
	virtual void refreshCUDAArray(){
		m_deltaWeightMap->refreshCUDAArray();
	}
	virtual void unRegisterCUDAArray(){
		m_deltaWeightMap->unRegisterCUDAArray();
	}

	virtual void registerCUDAArray(){
		m_deltaWeightMap->registerCUDAArray();
	}
private:
	int camWidth;
	int camHeight;
	int panoramaWidth;
	int panoramaHeight;

	// gl functions
	bool m_initialized;

	GPUPanoramaWeightMap* m_panoWeightMap;
	GPUDeltaWeightMap* m_deltaWeightMap;
};

#endif // GPUPANORAMAWEIGHT_H