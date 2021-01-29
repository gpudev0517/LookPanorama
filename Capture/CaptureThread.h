
#ifndef CAPTURETHREAD_H
#define CAPTURETHREAD_H


#include <thread>

// Qt
#include <QPixmap>
#include <QtCore/QTime>
#include <QtCore/QThread>


#include "SharedImageBuffer.h"
#include "Config.h"
#include "Structures.h"

#include "AudioThread.h"

class ImageBuffer;

class CaptureThread : public IAudioThread
{
    Q_OBJECT

    public:
		CaptureThread(SharedImageBuffer *sharedImageBuffer, int deviceNumber, D360::Capture::CaptureDomain capType, int width, int height);
		virtual ~CaptureThread();

		void	init();
        void	stop();
		void	playAndPause(bool isPause);
        bool	connect();
        bool	disconnect();
		bool	reconnect();
        bool	isConnected();
        int		getInputSourceWidth();
        int		getInputSourceHeight();

		virtual AudioInput * getMic();

		void setFrameRate( float fps )
		{
			m_captureFPS = fps;
		}

		void setExposure( float exp )
		{
			m_captureExp = exp;
		}

		void setGain( float gain )
		{
			m_captureGain = gain;
		}

		void snapshot(bool isCalibreate = false);

		void forceFinish();

		void waitForFinish();

		bool isFinished();

		int getDeviceNumber() { return m_deviceNumber; }

    private:
        void updateFPS( int );
		void waitCapture();
		void wakeCapture();

		QString m_Name;
        SharedImageBuffer *sharedImageBuffer;
        //VideoCapture cap;
		D360::Capture* cap;

		D360::Capture::CaptureDomain m_captureType;

        SharedImageBuffer::ImageDataPtr m_grabbedFrame;
        QTime t;
        QMutex doStopMutex;
		QMutex doPauseMutex;
        QQueue<int> fps;

		QMutex finishMutex;
		QWaitCondition finishWC;
		bool m_finished;
		bool m_isCanGrab;

		QMutex pauseWCMutex;
		QWaitCondition pauseWC;

		bool m_isReplay;

        struct ThreadStatisticsData statsData;
        bool doStop;		
		bool doPause;
		bool doSnapshot;
		bool doCalibrate;
        int captureTime;
        int sampleNumber;
        int fpsSum;
        int m_deviceNumber;
        int width;
        int height;

		float m_captureExp;
		float m_captureGain;
		float m_captureFPS;

    protected:
        void run();

	public slots:
			void process();

    signals:
        void updateStatisticsInGUI( struct ThreadStatisticsData );
		void finished(int type, QString msg, int id);
		void started(int type, QString msg, int id);
		void report(int type, QString msg, int id);
		void snapshoted(int id);
		void firstFrameCaptured(int);
};

#endif // CAPTURETHREAD_H
