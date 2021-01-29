#pragma once

#include <thread>

// Qt
#include <QtCore/QTime>
#include <QtCore/QThread>
// Local

#include "SharedImageBuffer.h"
#include "Config.h"
#include "Structures.h"

#include "CaptureLiDAR.h"
#include "StreamProcess.h"

class LiDARThread : public QObject
{
	Q_OBJECT
public:
	LiDARThread( QObject* main = NULL );
	~LiDARThread();

	// thread
	void	initialize( SharedImageBuffer *sharedImageBuffer);
	void	qquit();

	void	stopThread();
	void	waitForFinish();
	bool	isFinished();

	void	start();
	void	stop();
	void	pause();

	bool	connect();
	bool	disconnect();
	bool	isConnected();

	void startThread();

	virtual LiDARInput * getInputDevice();

private:
	void updateFPS( int );

	SharedImageBuffer *sharedImageBuffer;
	LiDARInput* cap;

	SharedImageBuffer::ImageDataPtr m_grabbedFrame;

	QThread* threadInstance;
	QTime t;

	QMutex doExitMutex;
	bool doExit;
	QMutex finishMutex;
	QWaitCondition finishWC;

	QMutex doPauseMutex;
	QWaitCondition pausehWC;

	QQueue<int> fps;
	StreamProcess* m_streamer;

	bool	m_finished;

	struct ThreadStatisticsData statsData;
	int captureTime;
	int sampleNumber;
	int fpsSum;
	int m_deviceNumber;
	QString m_Name;
	QObject* m_Main;

protected:
	void run();

public slots:
	void process();

signals:
	void updateStatisticsInGUI( struct ThreadStatisticsData );
	void finished( int type, QString msg, int id );
	void started( int type, QString msg, int id );

private:

};
