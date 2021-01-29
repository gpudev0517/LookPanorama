#pragma once

#define ENABLE_LOG 1

#include <QObject>
#include <QMutex>
#include <QRunnable>
#include <QDateTime>

#include <QOpenGLFunctions>
#include <QtGui/QOpenGLShaderProgram>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFunctions_2_0>
#include <QOpenGLFunctions_4_3_Compatibility>

#include <iostream>
#include <fstream>
#include <string>

#include "Structures.h"
#include "SharedImageBuffer.h"
#include "Config.h"

#include "GLSLColorCvt.h"
#include "GLSLGainCompensation.h"
#include "GLSLUnwarp.h"
#include "GLSLWeightMap.h"
#include "GLSLFinalPanorama.h"
#include "SingleViewProcess.h"
#include "SinglePanoramaUnit.h"

#ifdef USE_CUDA
#include "CUDAFinalPanorama.h"
#endif

#ifdef USE_CUDA
class DoStitchThread : public QThread
{
public:
	DoStitchThread();
	virtual ~DoStitchThread();

	void init();
	void run();
	bool m_bIsRunning;
private:
	QMutex mutex;
	QOpenGLContext* m_pContext;
};
#endif

class D360Stitcher: public QObject//, public QRunnable
{
	Q_OBJECT

public:
	D360Stitcher(SharedImageBuffer *sharedImageBuffer, QObject* main = NULL);
	virtual ~D360Stitcher(void);

	void init(QOpenGLContext* context);
	void buildOutputNode(int width, int height);

	void reset(); // this will be called to dispose any objects that were allocated in init()
	void removeOutputNode();
	void clear(); // clear all states. this is needed for configuration switch.
	void setup();
	void startThread();
	void stopStitcherThread();
	void playAndPause(bool isPause);
	void calcGain();
	void resetGain();
	void restitch(bool cameraParamChanged = false);
	void updateCameraParams();
	void updateWeightMapParams(WeightMapEyeMode eyeMode, int cameraIndex, int cameraIndex_,  float strength, int radius, float falloff, bool isIncrement, bool bChangeCameraIdx, int x = -1, int y = -1, bool isRightPos = false);

	bool checkAvailableIndex(int eyeIndex, int cameraIndex);
	int getAvailableIndex(int eyeIndex, int cameraIndex);

	QOpenGLContext* getContext();
	QOffscreenSurface* getSurface();
	GlobalAnimSettings* getGaSettings();

	int getRawTextureId(int cameraIndex);
	int getPanoramaTextureId();
	int getPanoramaTextureIdForInteract();

	GPUResourceHandle getPanoramaFBO();

	void snapshot();
	void initForReplay();

	void waitForFinish();
	void setFinished() { m_finished = true; };
	bool isFinished();

	void setWeightMapChanged(bool weightMapChanged);
	void resetEditedWeightMap();

	// for Mirror
	bool m_isLeft = false;

	struct ThreadStatisticsData getStatisticsData() { return statsData; }
private:
	GlobalAnimSettings* m_gaSettings;
	QThread* m_stitcherThreadInstance;
	SharedImageBuffer* sharedImageBuffer;

	std::map<int, ImageBufferData> cameraFrameRaw;
	std::map<int, ImageBufferData> cameraFrameUse;
	int cameraFrameProcN;
	bool removeBannerAll : 1;
	bool removeBannerLast : 1;
	int removeBannerIndex;
	QMutex m_stitchMutex;

	std::vector<BannerInfo> bannerInfos;
	QMutex m_bannerMutex;

	QObject* m_Main;
	QString m_Name;
	QOffscreenSurface* m_surface;
	QOpenGLContext* m_context;
	QOpenGLFunctions_2_0* functions_2_0;
	QOpenGLFunctions_4_3_Compatibility* functions_4_3;

	int m_nViewCount;

	// per camera
	std::vector<SingleViewProcess*> *m_viewProcessors;

	// Multi-Nodal
	QMap<int, GPUNodalInput**> m_nodalColorCvtMap;
	unsigned char* liveGrabBuffer;

	// per panorama
	std::vector<SinglePanoramaUnit*> *m_panoramaUnits;

	// final
	GPUPanorama** m_Panorama;
	GPUFinalPanorama** m_finalPanorama;

	QMutex finishMutex;
	QWaitCondition finishWC;
	bool m_finished;

	QMutex doPauseMutex;
	bool doPause;
	bool doCalcGain;
	bool doResetGain;
	bool doReStitch;
	bool doUpdateCameraParams;

	bool doSnapshot;

	unsigned char *m_fboBuffer;
	QMutex outputBufferMutex;

	// Texture id map for weight map textures
	std::vector<GPUResourceHandle> *srcWeightTextures;

	int m_nPanoramaReadyCount;

	bool m_isCameraParamChanged;
	int m_nSeamViewId;


	// weightmap
	std::vector<WeightMapUndoRedo> m_undoRedo;
	WEIGHTMAP_UNDOREDO_STATE m_undoRedoState;
	int m_undoRedoBufferIdx;
	bool m_isUndoRedo;
	bool m_weightmapCamIdxChangedByUndoRedo;

	bool m_weightMapChanged;
	bool m_isWeightMapReset;
	QMutex doWeightmapUpdateParamMutex;
	QMutex doWeightMapChangedMutex;
	QMutex doWeightMapSaveMutex;
	WeightMapPaintMode m_WeightmapPaintingMode;

	int m_weightMapCameraIndex;
	int m_weightMapCameraIndex_;
	int m_weightmapOriginalCameraIndex;
	WeightMapEyeMode m_weightMapEyeMode;
	float m_weightMapStrength;
	int m_weightMapRadius;
	float m_weightMapFallOff;
	bool m_isWeightMapIncrement;
	bool m_weightMapRightPos;
	bool m_weightmapCamIdxChanged;
	

	std::vector<vec2> m_weightMapEditPoints;

	// for Mirror
	std::vector<vec2> m_weightMapEditPoints_;

	QString m_panoWeightPath;
	bool m_panoWeightRecordOn;
	int m_panoWeightSkipFrames;

	//color temperature
	float m_colorTemperature;
	vec3 m_lightColor;

public:

	std::vector<SingleViewProcess *> *viewProcessers() {
		return m_viewProcessors;
	}

	void setLutData(QVariantList *vList);

	void setWeightmapPaintingMode(WeightMapPaintMode paintMode){
		m_WeightmapPaintingMode = paintMode;
		restitch();
	}
	void setWeightMapResetFlag(bool flag){ m_isWeightMapReset = flag; }
	void weightmapUndo(){ 
		m_isUndoRedo = true; 
		m_undoRedoBufferIdx--; 
		if (m_undoRedoBufferIdx < 0){
			m_undoRedoBufferIdx = 0;
		}
	}
	void weightmapRedo(){ 
		m_isUndoRedo = true; 
		m_undoRedoBufferIdx++; 
		if (m_undoRedoBufferIdx >= UNDOREDO_BUFFER_SIZE){
			m_undoRedoBufferIdx = UNDOREDO_BUFFER_SIZE - 1;
		}
		if (m_undoRedoBufferIdx >= m_undoRedo.size()){
			m_undoRedoBufferIdx = m_undoRedo.size() - 1;
		}
	}

	void setColorTemperature(float colorTemp){
		m_colorTemperature = colorTemp;
		restitch();
	}

	void copyFromUndoRedoBuffer(int panoUnitIdx, int weightmapCamIndex);
	void copyToUndoRedoBuffer(int panoUnitIdx, int weightmapCamIdx, WeightMapUndoRedo &undoRedo);
	void clampUndoRedoBufferSize();

	void deltaWeight2UndoRedo();
	void undoRedo2DeltaWeight();

	void sendUndoRedoUIChangedToMainWindow();

	float getColorTemperature() { return m_colorTemperature; }

	vec3 calculateLightColorWithTemperature(float inTemperature);

public slots:
	void process();
	void qquit();
	void updateStitchFrame(ImageBufferData& frame, int camIndex);
	void lockBannerMutex();
	void unlockBannerMutex();
	BannerInfo& createNewBanner();
	void cueRemoveBannerAll();
	void cueRemoveBannerLast();
	void cueRemoveBannerAtIndex(int index);
	BannerInfo* getBannerLast();
	BannerInfo* getBannerAtIndex(int index);
	void updateBannerVideoFrame(ImageBufferData& frame, int bannerId);
	void updateBannerImageFrame(QImage image, int bannerId);
	void saveBanner(GlobalAnimSettings& settings);
	void saveWeightMaps(QString iniPath);
	void recordMaskMap(QString maskPath);

	void setWeightMapEditMode(bool isEditMode);

	void DoCompositePanorama(GlobalAnimSettings* gasettings);

	void doCaptureIncomingFrames();
	
	void doStitch(bool stitchOn = true);
	void doMakePanorama();
	void doOutput();

protected:
	void run();	

	void initialize();

	void makeCameraWeightMap();

private:
	void calcExposure();
	void updateFPS(int);

	bool isCameraReadyToProcess(int camIndex);
	void processIndividualViews(bool weightMapChangedByCameraParam);
	void updateWeightmap(bool weightMapChanged);
	void stitchPanorama();

	void processSaveWeightMaps();

	void releaseStitcherThread();

	bool doStop;
	bool isFirstFrame;

	QTime t;
	QQueue<int> fps;
	QMutex doStopMutex;
	struct ThreadStatisticsData statsData;
	int fpsSum;
	int sampleNumber;

	bool m_bRunningByCuda;
	int m_cudaThreadCount;
	int m_idxCurrentGPU;

#ifdef USE_CUDA
	DoStitchThread *m_Thread;
	cudaStream_t m_stream[2];
#endif

signals:
	void newPanoramaFrameReady(unsigned char* buffer);

	void updateStatisticsInGUI(struct ThreadStatisticsData);
	void finished(int type, QString msg, int id);
	void started(int type, QString msg, int id);

	void snapshoted();
};

