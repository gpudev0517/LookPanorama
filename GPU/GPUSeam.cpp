#include "GPUSeam.h"
#include "define.h"
#include "common.h"
#include "QmlMainWindow.h"


#ifdef USE_CUDA
extern "C" void runSeamRegion_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int viewCnt, int viewIdx);
extern "C" void runSeamMask_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height);
extern "C" void runSeam_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, cudaTextureObject_t maskTex, int width, int height);
#endif //USE_CUDA

extern QmlMainWindow *g_mainWindow;

GPUSeam::GPUSeam(QObject *parent) : QObject(parent)
, m_initialized(false)
, m_showSeamIdx(-1)
{
#ifdef USE_CUDA
	if (g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		m_seamMask = new CUDASeamMask();
		m_seamRegion = new CUDASeamRegion();
	}
	else
#endif
	{
		m_seamMask = new GLSLSeamMask();
		m_seamRegion = new GLSLSeamRegion();
	}
	
	isSeamIdxChanged = false;
}

GPUSeam::~GPUSeam()
{
	if (m_initialized)
	{
		if (m_seamMask)
		{
			delete m_seamMask;
			m_seamMask = NULL;
		}
		if (m_seamRegion)
		{
			delete m_seamRegion;
			m_seamRegion = NULL;
		}
	}
}

void GPUSeam::setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0)
{
	m_seamRegion->setGL(gl, functions_2_0);
	m_seamMask->setGL(gl, functions_2_0);
}

void GPUSeam::initialize(int panoWidth, int panoHeight, int viewCount)
{
	this->m_panoramaWidth = panoWidth;
	this->m_panoramaHeight = panoHeight;
	this->m_viewCount = viewCount;

	m_seamRegion->initialize(panoWidth, panoHeight, viewCount);
	m_seamMask->initialize(panoWidth, panoHeight);

	m_initialized = true;
}

void GPUSeam::render(GPUResourceHandle *weighttextures, bool weightMapChanged)
{
	if (weightMapChanged)
	{
		m_seamRegion->render(weighttextures);
	}
	if (weightMapChanged || isSeamIdxChanged)
	{
		m_seamRegion->getCameraRegionImage(m_showSeamIdx);
		m_seamMask->getSeamMaskImage(m_seamRegion->getTargetGPUResource());
	}
	if (isSeamIdxChanged)
	{
		isSeamIdxChanged = false;
	}
}

GPUResourceHandle GPUSeam::getBoundaryTexture()
{
	return m_seamRegion->getBoundaryTexture();
}

int GPUSeam::getSeamTexture() {

	if (m_showSeamIdx < 0)
	{
		return -1;
	}
	else
	{
		return m_seamMask->getTargetGPUResource();
	}
}

void GPUSeam::setSeamIndex(int index)
{
	m_showSeamIdx = index;
	isSeamIdxChanged = true;
}

// GPUSeamRegion
GPUSeamRegion::GPUSeamRegion(QObject *parent) : GPUProgram(parent)
{
}

GPUSeamRegion::~GPUSeamRegion() {

    if (m_initialized)
	{
		delete m_boundary;
		m_boundary = NULL;
    }
}

void GPUSeamRegion::initialize(int panoWidth, int panoHeight, int viewCount)
{
	m_panoramaWidth = panoWidth;
	m_panoramaHeight = panoHeight;
	m_viewCount = viewCount;
}

void GPUSeamRegion::render(GPUResourceHandle *weighttextures) {
	m_boundary->render(weighttextures);
}

const int GPUSeamRegion::getWidth() {
	return m_panoramaWidth;
}

const int GPUSeamRegion::getHeight() {
	return m_panoramaHeight;
}



//GLSLSeamRegion
GLSLSeamRegion::GLSLSeamRegion(QObject *parent) : GPUSeamRegion(parent)
{
}

GLSLSeamRegion::~GLSLSeamRegion() 
{
}

void GLSLSeamRegion::initialize(int panoWidth, int panoHeight, int viewCount) {

	GPUSeamRegion::initialize(panoWidth, panoHeight, viewCount);

	m_boundary = new GLSLBoundary();
	m_boundary->setGL(m_gl, m_functions_2_0);
	m_boundary->initialize(panoWidth, panoHeight, viewCount);

	////////////////////////////////////////////////////////
	//initialize camera region shader
	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, m_panoramaWidth, m_panoramaHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "cameraRegion");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/cameraRegion.frag");
#endif
	m_program->link();
	m_program->bind();

	int textureIds[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	m_gl->glUniform1iv(m_program->uniformLocation(QString("textures")), 8, textureIds);

	m_program->setUniformValue("viewCnt", viewCount);
	m_program->release();

	m_initialized = true;
}

void GLSLSeamRegion::getCameraRegionImage(int seamIdx)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = m_panoramaWidth;
	float height = m_panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);

	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_boundary->getTargetGPUResource());

	m_program->setUniformValue("viewIdx", seamIdx);


	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = "cameraRegion.png";
	saveTexture(m_fboId, width, height, strSaveName, m_gl);
#endif 
}


#ifdef USE_CUDA
CUDASeamRegion::CUDASeamRegion(QObject *parent) : GPUSeamRegion(parent)
{
}

CUDASeamRegion::~CUDASeamRegion() {
}

void CUDASeamRegion::initialize(int panoWidth, int panoHeight, int viewCount) {

	GPUSeamRegion::initialize(panoWidth, panoHeight, viewCount);

	m_boundary = new CUDABoundary();
	m_boundary->setGL(m_gl, m_functions_2_0);
	m_boundary->initialize(panoWidth, panoHeight, viewCount);

	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&m_cudaTargetArray, &channelFormat, m_panoramaWidth, m_panoramaHeight, cudaArraySurfaceLoadStore);

	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);

	m_initialized = true;
}

void CUDASeamRegion::getCameraRegionImage(int seamIdx)
{
	runSeamRegion_Kernel(m_cudaTargetSurface, m_boundary->getTargetGPUResource(), m_panoramaWidth, m_panoramaHeight, m_viewCount, seamIdx);
	
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[m_panoramaWidth * m_panoramaHeight];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, m_panoramaWidth *m_panoramaHeight, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, m_panoramaWidth, m_panoramaHeight, QImage::Format_Grayscale8);
	img.save(QString("SeamRegion_") + QString::number(seamIdx) + ".png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
}
#endif //USE_CUDA


// GPUSeamMask
GPUSeamMask::GPUSeamMask(QObject *parent) : GPUProgram(parent)
{
}

GPUSeamMask::~GPUSeamMask()
{
    
}

const int GPUSeamMask::getWidth() {
	return m_panoramaWidth;
}

const int GPUSeamMask::getHeight() {
	return m_panoramaHeight;
}


//GLSLSeamMask
GLSLSeamMask::GLSLSeamMask(QObject *parent) : GPUSeamMask(parent)
{
}

GLSLSeamMask::~GLSLSeamMask()
{

}

void GLSLSeamMask::initialize(int panoWidth, int panoHeight)
{
	m_panoramaWidth = panoWidth;
	m_panoramaHeight = panoHeight;

	////////////////////////////////////////////////////////
	//initialize mask shader
	// frame buffer
	m_gl->glGenTextures(1, &m_fboTextureId);
	m_gl->glBindTexture(GL_TEXTURE_2D, m_fboTextureId);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_gl->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	m_gl->glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, panoWidth, panoHeight, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	// load textures and create framebuffers
	m_gl->glGenFramebuffers(1, &m_fboId);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);
	m_gl->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_fboTextureId, 0);
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// create fbo shader
	m_program = new QOpenGLShaderProgram();
#ifdef USE_SHADER_CODE
	ADD_SHADER_FROM_CODE(m_program, "vert", "stitcher");
	ADD_SHADER_FROM_CODE(m_program, "frag", "mask");
#else
	m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, "./Shaders/stitcher.vert");
	m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, "./Shaders/mask.frag");
#endif
	m_program->link();
	m_program->bind();

	m_program->setUniformValue("texture", 0);
	m_program->setUniformValue("width", panoWidth);
	m_program->setUniformValue("height", panoHeight);
	m_program->release();

	m_initialized = true;
}

void GLSLSeamMask::getSeamMaskImage(GPUResourceHandle regionTexture)
{
	m_program->bind();
	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, m_fboId);

	float width = m_panoramaWidth;
	float height = m_panoramaHeight;

	m_gl->glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	m_gl->glClear(GL_COLOR_BUFFER_BIT);
	m_gl->glViewport(0, 0, width, height);

	m_gl->glActiveTexture(GL_TEXTURE0);
	m_gl->glBindTexture(GL_TEXTURE_2D, regionTexture);

	m_gl->glDrawArrays(GL_TRIANGLES, 0, 3);

	m_gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	m_program->release();

#if DEBUG_SHADER
	QString strSaveName = "seamMask.png";
	saveTexture(m_fboId, width, height, strSaveName, m_gl);
#endif
}

#ifdef USE_CUDA

CUDASeamMask::CUDASeamMask(QObject *parent) : GPUSeamMask(parent)
{
}

CUDASeamMask::~CUDASeamMask()
{

}

void CUDASeamMask::initialize(int panoWidth, int panoHeight)
{
	m_panoramaWidth = panoWidth;
	m_panoramaHeight = panoHeight;

	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&m_cudaTargetArray, &channelFormat, m_panoramaWidth, m_panoramaHeight, cudaArraySurfaceLoadStore);


	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);

	m_initialized = true;
}

void CUDASeamMask::getSeamMaskImage(GPUResourceHandle regionTexture)
{
	runSeamMask_Kernel(m_cudaTargetSurface, regionTexture, m_panoramaWidth, m_panoramaHeight);
	
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[m_panoramaWidth * m_panoramaHeight];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, m_panoramaWidth *m_panoramaHeight, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, m_panoramaWidth, m_panoramaHeight, QImage::Format_Grayscale8);
	img.save(QString("seammask.png"));
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif

}
#endif //USE_CUDA