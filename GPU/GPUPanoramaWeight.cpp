#include "GPUPanoramaWeight.h"
#include "QmlMainWindow.h"

extern QmlMainWindow *g_mainWindow;

GPUPanoramaWeight::GPUPanoramaWeight(QObject *parent) : QObject(parent)
, m_initialized(false)
{
#ifdef USE_CUDA
	if (g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		m_panoWeightMap = new CUDAPanoramaWeightMap();
		m_deltaWeightMap = new CUDADeltaWeightMap();
	}
	else
#endif
	{
		m_panoWeightMap = new GLSLPanoramaWeightMap();
		m_deltaWeightMap = new GLSLDeltaWeightMap();
	}
	
	
}

GPUPanoramaWeight::~GPUPanoramaWeight()
{
	if (m_initialized)
	{
		delete m_panoWeightMap;
		delete m_deltaWeightMap;
	}
}

void GPUPanoramaWeight::setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3)
{
	m_panoWeightMap->setGL(gl, functions_2_0, functions_4_3);
	m_deltaWeightMap->setGL(gl, functions_2_0, functions_4_3);
}

void GPUPanoramaWeight::initialize(int camWidth, int camHeight, int panoWidth, int panoHeight)
{
	this->camWidth = camWidth;
	this->camHeight = camHeight;
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

	m_panoWeightMap->initialize(camWidth, camHeight, panoWidth, panoHeight);
	m_deltaWeightMap->initialize(camWidth, camHeight, panoWidth, panoHeight);

	m_initialized = true;
}

void GPUPanoramaWeight::renderWeight(GPUResourceHandle srcWeight, int camID)
{
	m_panoWeightMap->render(srcWeight, getDeltaWeightTexture(), camID);
}

void GPUPanoramaWeight::renderDelta(float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM, int camID)
{
	m_deltaWeightMap->render(radius, falloff, strength, centerx, centery, increment, globalM, camID);
}

void GPUPanoramaWeight::resetDeltaWeight()
{
	m_deltaWeightMap->resetMap();
}

void GPUPanoramaWeight::setCameraInput(CameraInput camInput)
{
	m_panoWeightMap->setCameraInput(camInput);
	m_deltaWeightMap->setCameraInput(camInput);
}

void GPUPanoramaWeight::updateCameraParams()
{
	m_panoWeightMap->updateCameraParams();
	m_deltaWeightMap->updateCameraParams();
}

void GPUPanoramaWeight::saveWeightmap(QString filename)
{
	m_deltaWeightMap->saveWeightmap(filename);
}

void GPUPanoramaWeight::loadWeightmap(QString filename)
{
	m_deltaWeightMap->loadWeightmap(filename);
}