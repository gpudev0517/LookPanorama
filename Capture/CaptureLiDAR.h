#ifndef CAPTURELIDAR_H
#define CAPTURELIDAR_H

#include <memory>
#include <QImage>
#include "Capture.h"
#include "QTHandler.h"
#include <iostream>
#include "SharedImageBuffer.h"
#include "RosDataConverter.h"

#define DEFAULT_BUFLEN 5120000


class LiDARInput : D360::Capture
{
public:

	LiDARInput( SharedImageBuffer *sharedImageBuffer ) :
		m_sharedImageBuffer( sharedImageBuffer )
	{
		m_captureDomain = D360::Capture::CAPTURE_LIDAR;
		init();
	}
	virtual ~LiDARInput()
	{
		if( m_RostoIMG )
			delete m_RostoIMG;
		close();
	}

	virtual void		reset( ImageBufferData& frame );
	virtual bool		open(int port);
	virtual void		close();
	virtual double		getProperty( int );
	virtual bool		setProperty( int, double );
	virtual bool		grabFrame( ImageBufferData& frame );
	virtual bool		retrieveFrame( int channel, ImageBufferData& frame );

private:
	void init();

	SharedImageBuffer *m_sharedImageBuffer;

	int m_port; // listening port

	int m_nWidth;
	int m_nHeight;

	char* pointCloudBuf;
	char* tempBuffer = new char[DEFAULT_BUFLEN];	//temporary buffer for merge buffer received Tcp
	int currentptr = 0; //position of tempbuffer in where recveived buffer is copied
	int nCloudBufSize = 0;
	bool bNewCloud = true;
	RosDataConverter* m_RostoIMG;
};

#endif //CAPTURELIDAR_H