#ifndef CUDABANNER_H
#define CUDABANNER_H

#ifdef USE_CUDA
#include "GPUProgram.h"
#include "GLSLColorCvt.h"

#include "GPUBanner.h"


/// <summary>
/// The opengl shader that adds each banner
/// </summary>
class CUDABill : public GPUBill
{
	Q_OBJECT
public:
	explicit CUDABill(QObject *parent = 0);
	virtual ~CUDABill();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void render(std::vector<BannerInfo*> bannerInputs);

protected:
	float *m_devPaiPlane;
	float *m_devHomography;
};

/// <summary>
/// The opengl shader that makes final banner with background
/// </summary>
class CUDABanner : public GPUBanner
{
	Q_OBJECT
public:
	explicit CUDABanner(QObject *parent = 0);
	virtual ~CUDABanner();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void render(GPUResourceHandle srcTextureId, GPUResourceHandle billTextureId);
};

#endif //USE_CUDA

#endif // CUDABANNER_H