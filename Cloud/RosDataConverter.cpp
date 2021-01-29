#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "RosDataConverter.h"
#include "common.h"
#include "define.h"

#define DEFAULT_PORT "8888"

// Need to link with Ws2_32.lib, Mswsock.lib, and Advapi32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")


RosDataConverter::RosDataConverter()
{

}

RosDataConverter::~RosDataConverter()
{
}

uint8_t* RosDataConverter::convertToImageFormPCL( char* cloudBuf )
{
	m_pointCloud.deserialize( (unsigned char*)cloudBuf );

	int nCloudPtCnt = m_pointCloud.height * m_pointCloud.width;
	CloudPoint* points = (CloudPoint*)malloc( nCloudPtCnt * sizeof( CloudPoint ) );
	memcpy( &points[0], m_pointCloud.data, nCloudPtCnt * sizeof( CloudPoint ) );

	uint8_t* pclImage = new uint8_t[LIDAR_STREAM_HEIGHT*LIDAR_STREAM_WIDTH];
	memset( pclImage, 128, LIDAR_STREAM_HEIGHT * LIDAR_STREAM_WIDTH );
	for (int i = 0; i < LIDAR_BAND_COUNT; i++ )
	{
		memset( pclImage + i * LIDER_BAND_HEIGHT*LIDAR_STREAM_WIDTH, 255, LIDAR_HEIGHT*LIDAR_STREAM_WIDTH );
	}
	for (int i = 0; i < nCloudPtCnt ; i++)
	{
		float x = points[i].x;
		float y = points[i].y;
		float z = points[i].z;
		uint8_t ring = points[i].ring;
		if(ring > 63)
			continue;
		float length = sqrtf( x*x + y*y + z*z );
		if( length < 0.1 )
			continue;
		if( length >= LIDAR_BAND_COUNT * LIDAR_STREAM_MAX_VALUE * 1.2 / 100 )
			continue;

		int rowNum = ring;
		int colNum = 0;

		float angle_x = atan2f( y, x )*180/M_PI + 180;
		colNum = angle_x * 1024 / 360;
		int serialDepth = floor( length * 100 / LIDAR_RANGE_RESOLUTION );//m -> cm
		int bandNum = serialDepth / LIDAR_STREAM_MAX_VALUE;
		pclImage[bandNum* LIDAR_STREAM_WIDTH* LIDER_BAND_HEIGHT + rowNum * LIDAR_STREAM_WIDTH + colNum] = (uint8_t)(serialDepth - bandNum * LIDAR_STREAM_MAX_VALUE );
	}
	free( cloudBuf );
	free( points );
	return pclImage;
}

void RosDataConverter::getCloudFileds(const sensor_msgs::PointCloud2& ptCloud)
{
	int nCloudPtCnt = ptCloud.height * ptCloud.width;
	CloudPoint* points = (CloudPoint*)malloc( nCloudPtCnt * sizeof( CloudPoint ) );
	memcpy( &points[0], ptCloud.data, nCloudPtCnt * sizeof( CloudPoint ) );
}