#include "GPUPanoramaPostProcessing.h"
#include "define.h"

GPUPanoramaPostProcessing::GPUPanoramaPostProcessing(QObject *parent) : GPUProgram(parent)
{
}

GPUPanoramaPostProcessing::~GPUPanoramaPostProcessing()
{
}

const int GPUPanoramaPostProcessing::getWidth()
{
	return panoramaWidth;
}

const int GPUPanoramaPostProcessing::getHeight()
{
	return panoramaHeight;
}