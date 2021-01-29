#include "GPUBanner.h"
#include "define.h"
#include "QmlMainWindow.h"

#include "GLSLBanner.h"
#ifdef USE_CUDA
#include "CUDABanner.h"
#endif

// BannerRenderer

extern QmlMainWindow *g_mainWindow;
BannerRenderer::BannerRenderer(QObject *parent) : QObject(parent)
{
#ifdef USE_CUDA
	if (g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		m_bill = new CUDABill();
		m_banner = new CUDABanner();
	}
	else
#endif
	{
		m_bill = new GLSLBill();
		m_banner = new GLSLBanner();
	}
	
	//m_bannerSmooth = new GLSLAntialiasing();
}

BannerRenderer::~BannerRenderer()
{
	delete m_bill;
	delete m_banner;
	//delete m_bannerSmooth;
}

void BannerRenderer::setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0)
{
	m_bill->setGL(gl, functions_2_0);
	m_banner->setGL(gl, functions_2_0);
	//m_bannerSmooth->setGL(gl, functions_2_0);
}

void BannerRenderer::initialize(int panoWidth, int panoHeight)
{
	m_bill->initialize(panoWidth, panoHeight);
	m_banner->initialize(panoWidth, panoHeight);
	//m_bannerSmooth->initialize(panoWidth, panoHeight);
	//m_bannerSmooth->setJitter(8);
}

void BannerRenderer::render(GPUResourceHandle srcTextureId, std::vector<BannerInfo*> bannerInputs)
{
	m_bill->render(bannerInputs);
	GPUResourceHandle billTextureId = m_bill->getTargetGPUResource();
	{
		//m_bannerSmooth->render(billTextureId);
		//billTextureId = m_bannerSmooth->getTargetGPUResource();
	}
	m_banner->render(srcTextureId, billTextureId);
}

GPUResourceHandle BannerRenderer::getBannerTexture()
{
	return m_banner->getTargetGPUResource();
}

// GLSLBill
GPUBill::GPUBill(QObject *parent) : GPUProgram(parent)
{
}

GPUBill::~GPUBill()
{
}

const int GPUBill::getWidth()
{
	return panoramaWidth;
}

const int GPUBill::getHeight()
{
	return panoramaHeight;
}

// GLSLBanner
GPUBanner::GPUBanner(QObject *parent) : GPUProgram(parent)
{
}

GPUBanner::~GPUBanner()
{
}

const int GPUBanner::getWidth()
{
	return panoramaWidth;
}

const int GPUBanner::getHeight()
{
	return panoramaHeight;
}