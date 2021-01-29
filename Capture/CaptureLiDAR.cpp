#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "CaptureLiDAR.h"
#include "CaptureProp.h"

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


SOCKET ConnectSocket;

void LiDARInput::init()
{
	m_nWidth = 0;
	m_nHeight = 0;
	m_RostoIMG = new RosDataConverter();
	return;
}


void LiDARInput::reset( ImageBufferData& frame )
{

}

bool LiDARInput::open( int port )
{
	m_port = port;

	ConnectSocket = INVALID_SOCKET;

	WSADATA wsaData;
	// Initialize Winsock
	int iResult = WSAStartup( MAKEWORD( 2, 2 ), &wsaData );
	if( iResult != 0 )
	{
		printf( "WSAStartup failed with error: %d\n", iResult );
		return false;
	}

	struct addrinfo *result = NULL;
	struct addrinfo *ptr = NULL;
	struct addrinfo hints;

	ZeroMemory( &hints, sizeof( hints ) );
	hints.ai_family = AF_UNSPEC;
	hints.ai_socktype = SOCK_STREAM;
	hints.ai_protocol = IPPROTO_TCP;

	char portStr[10];
	itoa( m_port, portStr, 10 );
	// Resolve the server address and port
	iResult = getaddrinfo( "10.70.3.75", portStr, &hints, &result );
	if( iResult != 0 )
	{
		printf( "getaddrinfo failed with error: %d\n", iResult );
		WSACleanup();
		return false;
	}

	// Attempt to connect to an address until one succeeds
	for( ptr = result; ptr != NULL; ptr = ptr->ai_next )
	{

		// Create a SOCKET for connecting to server
		ConnectSocket = socket( ptr->ai_family, ptr->ai_socktype,
								ptr->ai_protocol );
		if( ConnectSocket == INVALID_SOCKET )
		{
			printf( "socket failed with error: %ld\n", WSAGetLastError() );
			WSACleanup();
			return false;
		}

		// Connect to server.
		iResult = connect( ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen );
		if( iResult == SOCKET_ERROR )
		{
			closesocket( ConnectSocket );
			ConnectSocket = INVALID_SOCKET;
			continue;
		}
		break;
	}

	freeaddrinfo( result );
	// open tcp listening here
	return true;
}


bool LiDARInput::grabFrame( ImageBufferData& frame )
{
	char* recvBuf = new char[DEFAULT_BUFLEN];
	int nResult = recv( ConnectSocket, recvBuf, DEFAULT_BUFLEN, 0 );
	if( nResult < 0 )
	{
		printf( "recv failed with error: %d\n", WSAGetLastError() );
		free( recvBuf );
		return false;
	}
	else if( nResult == 0 )
	{
		printf( "Connection closed\n" );
		free( recvBuf );
		return false;
	}

	memcpy( tempBuffer + currentptr, recvBuf, nResult );
	free( recvBuf );
	recvBuf = NULL;

	currentptr += nResult;

	if( bNewCloud )
	{
		if( currentptr < 4 )
			return false;

		nCloudBufSize = ( (uint32_t)( *( (unsigned char*)tempBuffer ) ) );
		nCloudBufSize |= ( (uint32_t)( *(unsigned char*)( tempBuffer + 1 ) ) ) << ( 8 * 1 );
		nCloudBufSize |= ( (uint32_t)( *(unsigned char*)( tempBuffer + 2 ) ) ) << ( 8 * 2 );
		nCloudBufSize |= ( (uint32_t)( *(unsigned char*)( tempBuffer + 3 ) ) ) << ( 8 * 3 );
		bNewCloud = false;
		return false;
	}
	else
	{
		if( currentptr < nCloudBufSize + 4 )
			return false;
		pointCloudBuf = new char[nCloudBufSize];
		char* extraBuf = new char[currentptr - nCloudBufSize - 4];
		memcpy( pointCloudBuf, tempBuffer + 4, nCloudBufSize );
		memcpy( extraBuf, tempBuffer + nCloudBufSize + 4, currentptr - nCloudBufSize - 4 );
		memcpy( tempBuffer, extraBuf, currentptr - nCloudBufSize - 4 );
		currentptr = currentptr - nCloudBufSize - 4;
		bNewCloud = true;

		free( extraBuf );
	}

	return true;
}

bool LiDARInput::retrieveFrame( int, ImageBufferData& frame )
{

	uint8_t *srcPointer = m_RostoIMG->convertToImageFormPCL( pointCloudBuf);

	frame.mFormat = ImageBufferData::LIDAR;

	frame.mImageY.width = LIDAR_STREAM_WIDTH;
	frame.mImageY.height = LIDAR_STREAM_HEIGHT;
	frame.mImageY.stride = LIDAR_STREAM_WIDTH;
	if( frame.mImageY.buffer == NULL )
	{
		frame.mImageY.makeBuffer( frame.mImageY.stride * frame.mImageY.height );
	}
	memcpy( frame.mImageY.buffer, srcPointer, frame.mImageY.stride * frame.mImageY.height );

	frame.mImageU = AlignedImage();
	frame.mImageV = AlignedImage();

	free( srcPointer );

	return true;
}

void LiDARInput::close()
{
	closesocket( ConnectSocket );
	WSACleanup();
}

double LiDARInput::getProperty( int property_id )
{
	//if( m_pVideoCodecCtx == NULL )
	//	return 0;

	int ival = 0;
	float fval = 0;

	switch( property_id )
	{
		// OCV parameters
		case CV_CAP_PROP_FRAME_WIDTH: return 1024;
		case CV_CAP_PROP_FRAME_HEIGHT: return 64;
	}

	return 0.0f;
}

bool LiDARInput::setProperty( int property_id, double value )
{
	switch( property_id )
	{
		case CV_CAP_PROP_FRAME_WIDTH:
			m_nWidth = (int)value;
			break;
		case CV_CAP_PROP_FRAME_HEIGHT:
			m_nHeight = (int)value;
			break;
	}

	return true;
}