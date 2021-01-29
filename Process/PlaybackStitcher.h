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

class PlaybackStitcher: public QObject//, public QRunnable
{
	Q_OBJECT

public:
	PlaybackStitcher(SharedImageBuffer *sharedImageBuffer, QObject* main = NULL);
	virtual ~PlaybackStitcher(void);

	void init(QOpenGLContext* context);

	void reset(); // this will be called to dispose any objects that were allocated in init()
	void clear(); // clear all states. this is needed for configuration switch.
	void setup();
	void startThread();
	void stopStitcherThread();
	void playAndPause(bool isPause);
	void setSeekFrames(int nFrames, bool bIsFirstFrame = false);

	int getPanoramaTextureId();

	void initForReplay();

	void waitForFinish();
	void setFinished() { m_finished = true; };
	bool isFinished();

	struct ThreadStatisticsData getStatisticsData() { return statsData; }
private:
	GlobalAnimSettings* m_gaSettings;
	QThread* m_stitcherThreadInstance;
	SharedImageBuffer* sharedImageBuffer;

	ImageBufferData playbackFrameRaw;
	int frameProcN;

	QMutex m_stitchMutex;

	QObject* m_Main;
	QString m_Name;
	QOffscreenSurface* m_surface;
	QOpenGLContext* m_context;
	QOpenGLFunctions_2_0* functions_2_0;
	QOpenGLFunctions_4_3_Compatibility* functions_4_3;

	// per camera
	SingleViewProcess *m_viewProcessors;

	// final
	GPUPanorama* m_Panorama;

	QMutex finishMutex;
	QWaitCondition finishWC;
	bool m_finished;

	QMutex doPauseMutex;
	QMutex doSeekMutex;
	bool doPause;
	
public:

public slots:
	void process();
	void qquit();
	void updateStitchFrame(ImageBufferData& frame);

	void DoCompositePanorama(GlobalAnimSettings* gasettings);

	void doCaptureIncomingFrames();

protected:
	void run();	

	void initialize();

private:
	void updateFPS(int);

	void stitchPanorama();

	void releaseStitcherThread();

	bool doStop;
	bool isFirstFrame;

	QTime t;
	QQueue<int> fps;
	QMutex doStopMutex;
	struct ThreadStatisticsData statsData;
	int fpsSum;
	int sampleNumber;

signals:
	void newPanoramaFrameReady(unsigned char* buffer);

	void updateStatisticsInGUI(struct ThreadStatisticsData);
	void finished(int type, QString msg, int id);
	void started(int type, QString msg, int id);
};

