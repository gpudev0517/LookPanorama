#ifndef GPUBANNER_H
#define GPUBANNER_H

#include "GPUProgram.h"
#include "GLSLColorCvt.h"

class GPUBill;
class GPUBanner;

struct BannerInfo
{
	// ini info
	QString filePath;
	vec2 quad[4]; // window2pano_1x1
	// temp variable
	bool isValid = false; // UniColorCvt::DynRender() is true ?
	//
	ImageBufferData frame;
	mat4 paiPlane; // columns 0(ux), 1(uy), 2(uz), 3(org)
	float paiZdotOrg; // uz dot org
	mat3 homography; // H matrix for paiPlane->banner texture
	GPUUniColorCvt* billColorCvt = NULL;
	bool isStereoRight = false;
	bool isVideo = false;
	int id;
	static int seedId;

	BannerInfo();
	void dispose();
};

class BannerRenderer : public QObject
{
	Q_OBJECT
public:
	BannerRenderer(QObject *parent = 0);
	virtual ~BannerRenderer();

	void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0);
	void initialize(int panoWidth, int panoHeight);
	void render(GPUResourceHandle srcTextureId, std::vector<BannerInfo*> bannerInputs);

	GPUResourceHandle getBannerTexture();

private:
	GPUBill* m_bill;
	GPUBanner* m_banner;
};

/// <summary>
/// The opengl shader that adds each banner
/// </summary>
class GPUBill : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUBill(QObject *parent = 0);
	virtual ~GPUBill();

	virtual void initialize(int panoWidth, int panoHeight) = 0;
	virtual void render(std::vector<BannerInfo*> bannerInputs) = 0;

	virtual const int getWidth();
	virtual const int getHeight();

protected:
	int panoramaWidth;
	int panoramaHeight;
};

/// <summary>
/// The opengl shader that makes final banner with background
/// </summary>
class GPUBanner : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUBanner(QObject *parent = 0);
	virtual ~GPUBanner();

	virtual void initialize(int panoWidth, int panoHeight) = 0;
	virtual void render(GPUResourceHandle srcTextureId, GPUResourceHandle billTextureId) = 0;

	virtual const int getWidth();
	virtual const int getHeight();

protected:
	int panoramaWidth;
	int panoramaHeight;
};

#endif // GPUBANNER_H