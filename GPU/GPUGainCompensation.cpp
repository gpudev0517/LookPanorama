#include "GPUGainCompensation.h"
#include "common.h"
#include "define.h"

GPUGainCompensation::GPUGainCompensation(QObject *parent) : GPUProgram(parent)
{
}

GPUGainCompensation::~GPUGainCompensation()
{
}

int GPUGainCompensation::getReadyGPUResourceIndex()
{
	int textureIndex = m_workingGPUResource - 1;
	if (textureIndex < 0)
		textureIndex = textureIndex + m_targetCount;
	return textureIndex;
}

const int GPUGainCompensation::getWidth()
{
	return cameraWidth;
}

const int GPUGainCompensation::getHeight()
{
	return cameraHeight;
}