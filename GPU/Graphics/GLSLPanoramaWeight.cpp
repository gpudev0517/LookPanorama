#include "GLSLPanoramaWeight.h"

GLSLPanoramaWeight::GLSLPanoramaWeight(QObject *parent) : QObject(parent)
, m_initialized(false)
{
	m_panoWeightMap = new GLSLPanoramaWeightMap();
	m_deltaWeightMap = new GLSLDeltaWeightMap();
}

GLSLPanoramaWeight::~GLSLPanoramaWeight()
{
	if (m_initialized)
	{
		delete m_panoWeightMap;
		delete m_deltaWeightMap;
	}
}

void GLSLPanoramaWeight::setGL(QOpenGLFunctions* gl)
{
	m_gl = gl;

	m_panoWeightMap->setGL(gl);
	m_deltaWeightMap->setGL(gl);
}

void GLSLPanoramaWeight::initialize(int camWidth, int camHeight, int panoWidth, int panoHeight)
{
	this->camWidth = camWidth;
	this->camHeight = camHeight;
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

	m_panoWeightMap->initialize(camWidth, camHeight, panoWidth, panoHeight);
	m_deltaWeightMap->initialize(camWidth, camHeight, panoWidth, panoHeight);

	m_initialized = true;
}

void GLSLPanoramaWeight::renderWeight(GLuint srcWeight, int camID)
{
	m_panoWeightMap->render(srcWeight, getDeltaWeightTexture(), camID);
}

void GLSLPanoramaWeight::renderDelta(float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM, int camID)
{
	m_deltaWeightMap->render(radius, falloff, strength, centerx, centery, increment, globalM, camID);
}

void GLSLPanoramaWeight::resetDeltaWeight()
{
	m_deltaWeightMap->resetMap();
}

void GLSLPanoramaWeight::setCameraInput(CameraInput camInput)
{
	m_panoWeightMap->setCameraInput(camInput);
	m_deltaWeightMap->setCameraInput(camInput);
}

void GLSLPanoramaWeight::updateCameraParams()
{
	m_panoWeightMap->updateCameraParams();
	m_deltaWeightMap->updateCameraParams();
}

void GLSLPanoramaWeight::saveWeightmap(QString filename)
{
	m_deltaWeightMap->saveWeightmap(filename);
}

void GLSLPanoramaWeight::loadWeightmap(QString filename)
{
	m_deltaWeightMap->loadWeightmap(filename);
}