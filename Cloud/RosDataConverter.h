#ifndef ROSDATACONVERTER_H
#define ROSDATACONVERTER_H

#include "PointCloud2.h"

#define DEFAULT_BUFLEN 5120000


struct CloudPoint
{
	float x;
	float y;
	float z;
	float dummy1;
	float t;
	uint16_t reflectivity;
	uint16_t intensity;
	uint8_t ring;
	uint8_t dummy2, dummy3, dummy4;
	float dummy5;
};

class RosDataConverter
{
public:
	void init();
	RosDataConverter();
	~RosDataConverter();
	uint8_t* convertToImageFormPCL( char* cloudBuf);
	void getCloudFileds(const sensor_msgs::PointCloud2& ptCloud);
private:
	char* m_server;
	sensor_msgs::PointCloud2 m_pointCloud;
};

 
#endif //ROSDATACONVERTER_H