#include "GPUBoundary.h"
#include "define.h"
#include "common.h"

GPUBoundary::GPUBoundary(QObject *parent) : GPUProgram(parent)
{
}

GPUBoundary::~GPUBoundary()
{
}


const int GPUBoundary::getWidth()
{
	return panoramaWidth;
}

const int GPUBoundary::getHeight()
{
	return panoramaHeight;
}