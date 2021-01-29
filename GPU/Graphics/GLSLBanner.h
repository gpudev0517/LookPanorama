#ifndef GLSLBANNER_H
#define GLSLBANNER_H

#include "GPUProgram.h"
#include "GLSLColorCvt.h"
#include "GPUBanner.h"

/// <summary>
/// The opengl shader that adds each banner
/// </summary>
class GLSLBill : public GPUBill
{
	Q_OBJECT
public:
	explicit GLSLBill(QObject *parent = 0);
	virtual ~GLSLBill();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void render(std::vector<BannerInfo*> bannerInputs);

};

/// <summary>
/// The opengl shader that makes final banner with background
/// </summary>
class GLSLBanner : public GPUBanner
{
	Q_OBJECT
public:
	explicit GLSLBanner(QObject *parent = 0);
	virtual ~GLSLBanner();

	virtual void initialize(int panoWidth, int panoHeight);
	virtual void render(GPUResourceHandle srcTextureId, GPUResourceHandle billTextureId);
};

#endif // GLSLBANNER_H