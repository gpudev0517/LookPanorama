
#ifndef BannerThread_H
#define BannerThread_H


#include <thread>

// Qt
#include <QPixmap>
#include <QtCore/QTime>
#include <QtCore/QThread>

// Local
#include "CaptureImageFile.h"
#include "CaptureDShow.h"

#include "SharedImageBuffer.h"
#include "Config.h"
#include "Structures.h"

class ImageBuffer;

class BannerThread : public QThread
{
    Q_OBJECT
public:
	BannerThread(SharedImageBuffer *sharedImageBuffer, int bannerId, QString fileName, int width, int height, float fps);
	virtual ~BannerThread();

	void	init();
	void	stopBannerThread();
	void	playAndPause(bool isPause);
    bool	connect();
	bool	reconnect();
    bool	disconnect();
    bool	isConnected();
    int		getInputSourceWidth();
    int		getInputSourceHeight();
		
	void waitForFinish();
	void setFinished() { m_finished = true; };
	bool isFinished();

	int getBannerId() const { return bannerId; }

private:
	QString m_Name;
	QString m_fileName;

	D360::Capture* cap;
	D360::Capture::CaptureDomain m_captureType;

	SharedImageBuffer *sharedImageBuffer;
	SharedImageBuffer::ImageDataPtr m_grabbedFrame;

	QTime t;
    QMutex doStopMutex;
	QMutex doPauseMutex;
    QQueue<int> fps;

	QMutex finishMutex;
	QWaitCondition finishWC;
	bool m_finished;
	bool m_isCanGrab;

	bool m_isReplay;

	struct ThreadStatisticsData statsData;

	bool doStop;		
	bool doPause;
    int fpsSum;
    int bannerId; // m_deviceNumber
    int m_width;
    int m_height;
	float m_captureFPS;

protected:
    void run();

public slots:
	void process();
	void forceFinish();

signals:
	void newFrame( const QPixmap &frame, int frameNum );
	void finished(int type, QString msg, int id);
	void started(int type, QString msg, int id);
	void report(int type, QString msg, int id);
};

#endif // BannerThread_H
