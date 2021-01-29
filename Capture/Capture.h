#pragma once

#include <iostream>

#include <QImage>

#include "ImageBuffer.h"
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;
using namespace std;

namespace D360
{

	class Capture
	{
	public:

		Capture()
		{
			isSnapshot = false;
			init();
		}
		virtual ~Capture() { close(); }

		virtual bool   open(int index);
		virtual bool   start()
		{
			return true;
		}
		virtual void   close();
		virtual double getProperty(int);
		virtual bool   setProperty(int, double);
		virtual void   reset(ImageBufferData& frame)
		{

		}

		virtual bool grabFrame(ImageBufferData& frame);
		virtual bool retrieveFrame(int, ImageBufferData& frame);
		virtual void* retrieveAudioFrame();
		virtual void seekFirst();
		virtual void seekFrames(int nFrames);

		void setSnapshotPath(QString path)
		{
			m_snapshot = path;
		}
		void snapshot() { isSnapshot = true; }

		void setCurFrame(float curFrame)
		{
			m_curFrame = curFrame;
		}

		float getCurFrame()
		{
			return m_curFrame;
		}

		enum CaptureDomain
		{
			CAPTURE_DSHOW = 0,
			CAPTURE_VIDEO = 1,
            CAPTURE_FILE  = 2,
            CAPTURE_PTGREY = 3,
			CAPTURE_LIDAR = 4
		};
		virtual int getCaptureDomain()
		{
			return m_captureDomain;
		}
		virtual int msToWait()
		{
			return 0;
		}

		enum IncomingFrameType
		{
			None,
			Video,
			Audio
		};

		virtual IncomingFrameType getIncomingType()
		{
			return m_incomingType;
		}

		virtual ImageBufferData convertToRGB888(ImageBufferData& image) {
			return image;
		}

		virtual bool isOfflineMode()
		{
			switch (m_captureDomain)
			{
			case CAPTURE_DSHOW:
				return false;
			default:
				return true;
			}
		}

	protected:
		QString m_snapshot;
		CaptureDomain m_captureDomain;
		IncomingFrameType m_incomingType;

		float m_curFrame;

		void init();
		void errMsg(const char* msg, int errNum);

		int  getBpp();

		bool isSnapshot;
	};

}