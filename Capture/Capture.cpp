#include "Capture.h"

#include <iostream>

namespace D360 {

	//
	// Enumerate connected devices
	//

	void Capture::init()
	{
	}



	// Initialize camera input
	bool Capture::open(int wIndex)
	{
		return true;
	}


	void Capture::close()
	{
		return;
	}

	bool Capture::grabFrame(ImageBufferData& frame)
	{
		return true;
	}

	bool Capture::retrieveFrame(int channel, ImageBufferData& frame)
	{
		if (m_incomingType == D360::Capture::Video && m_captureDomain == CAPTURE_FILE)
			m_incomingType = D360::Capture::None;
		return true;
	}

	void Capture::seekFirst()
	{
	}

	void Capture::seekFrames(int nFrames)
	{
	}

	void* Capture::retrieveAudioFrame()
	{
		return NULL;
	}

	double Capture::getProperty(int property_id)
	{
		return 0.0f;
	}

	bool Capture::setProperty(int property_id, double value)
	{
		return true;
	}

	void Capture::errMsg(const char* msg, int errNum)
	{
		fprintf(stderr, "%s : %d\n", msg, errNum);
	}

	int  Capture::getBpp()
	{
		return 8;
	}
}
