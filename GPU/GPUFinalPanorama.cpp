#include "GPUFinalPanorama.h"
#include "define.h"
#include "common.h"

GPUPanorama::GPUPanorama(QObject *parent) : GPUProgram(parent)
{
}

GPUPanorama::~GPUPanorama()
{
}

const int GPUPanorama::getWidth()
{
	return panoramaViewWidth;
}

const int GPUPanorama::getHeight()
{
	return panoramaViewHeight;
}

// Final Panorama
GPUFinalPanorama::GPUFinalPanorama(QObject *parent) : GPUProgram(parent)
{
}

GPUFinalPanorama::~GPUFinalPanorama()
{
}

void GPUFinalPanorama::requestReconfig(int panoWidth, int panoHeight, bool isStereo)
{
	FinalPanoramaConfig config(panoWidth, panoHeight, isStereo);
	newConfig.push_back(config);
}

const int GPUFinalPanorama::getWidth()
{
	return panoramaViewWidth;
}

const int GPUFinalPanorama::getHeight()
{
	return panoramaViewHeight;
}