#include "SingleViewProcess.h"
#include "QmlMainWindow.h"

extern QmlMainWindow* g_mainWindow;
//extern QThread* g_mainThread;

SingleViewProcess::SingleViewProcess(SharedImageBuffer *sharedImageBuffer,
	int  deviceNumber,
	int  width, int height) :
sharedImageBuffer(sharedImageBuffer)
{
	this->camIndex = deviceNumber;
	this->width = width;
	this->height = height;

	init();
}

SingleViewProcess::~SingleViewProcess()
{
	clear();
	destroy();
}

void SingleViewProcess::init() {
	//doStop = false;

	m_paramChanged = true;
}

void SingleViewProcess::clear()
{
}

void SingleViewProcess::create(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3)
{
	/*m_surface = new QOffscreenSurface();
	m_surface->create();

	m_context = new QOpenGLContext();
	QSurfaceFormat format = m_surface->requestedFormat();
	format.setSwapInterval(0);
	format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
	m_context->setFormat(format);
	m_context->setShareContext(QOpenGLContext::globalShareContext());
	m_context->create();

	m_context->makeCurrent(m_surface);

	QOpenGLVersionProfile profile;
	profile.setProfile(m_context->surface()->format().profile());
	functions_2_0 = ((QOpenGLFunctions_2_0*)m_context->versionFunctions(profile));*/



	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();
	CameraInput& curCameraInput = gasettings->getCameraInput(camIndex);
	int imgWidth = curCameraInput.xres;
	int imgHeight = curCameraInput.yres;
	int pixel_format = curCameraInput.pixel_format;

	m_2rgbColorCvt = GPUColorCvt_2RGBA::createColorCvt(pixel_format, g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/);
	m_2rgbColorCvt->setGL(gl);

#ifdef USE_CUDA
	if (g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		m_gainCompensation = new CUDAGainCompensation();
	}
	else
#endif
	{
		m_gainCompensation = new GLSLGainCompensation();
	}
	
	m_gainCompensation->setGL(gl, functions_2_0);

#ifdef USE_CUDA
	if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/ ){
		m_camWeightMap = new CUDACameraWeightMap();
	}
	else
#endif
	{
		m_camWeightMap = new GLSLCameraWeightMap();
	}

	
	m_camWeightMap->setCameraInput(curCameraInput);
	m_camWeightMap->setGL(gl);

#ifdef USE_CUDA
	if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		m_unwarp = new CUDAUnwarp();
	}
	else
#endif
	{
		m_unwarp = new GLSLUnwarp();
	}

	
	m_unwarp->setCameraInput(curCameraInput);
	m_unwarp->setGL(gl, 0, functions_4_3);


	////
	int panoWidth = gasettings->m_panoXRes;
	int panoHeight = gasettings->m_panoYRes;

	m_2rgbColorCvt->initialize(imgWidth, imgHeight);
	m_gainCompensation->initialize(imgWidth, imgHeight);
	m_camWeightMap->initialize(imgWidth, imgHeight);
	m_unwarp->initialize(camIndex, imgWidth, imgHeight, panoWidth, panoHeight);

	//std::shared_ptr< D360Stitcher > stitcher = sharedImageBuffer->getStitcher();
	//stitcher->initImgProc(QOpenGLContext::globalShareContext());
}

void SingleViewProcess::destroy()
{
	//if (m_surface)
	{
		/*m_context->doneCurrent();
		m_context->moveToThread(g_mainThread);
		if (m_context != QOpenGLContext::currentContext())
		m_context->makeCurrent(m_surface);*/
		if (m_2rgbColorCvt)
		{
			delete m_2rgbColorCvt;
			m_2rgbColorCvt = NULL;
		}
		if (m_gainCompensation)
		{
			delete m_gainCompensation;
			m_gainCompensation = NULL;
		}
		if (m_unwarp)
		{
			delete m_unwarp;
			m_unwarp = NULL;
		}
		if (m_camWeightMap)
		{
			delete m_camWeightMap;
			m_camWeightMap = NULL;
		}

		/*m_surface->destroy();
		m_context->doneCurrent();
		delete m_context;
		m_context = NULL;
		delete m_surface;
		m_surface = NULL;*/
	}
}

GPUResourceHandle SingleViewProcess::getUnwarpTexture()
{
	return m_unwarp->getTargetGPUResource();
}

GPUResourceHandle SingleViewProcess::getTargetFrameBuffer()
{
	return m_unwarp->getTargetBuffer();
}

GPUResourceHandle SingleViewProcess::getRawTexture()
{
	return m_gainCompensation->getRawGPUResource();
}

GPUResourceHandle SingleViewProcess::getWeightTexture()
{
	return m_camWeightMap->getTargetGPUResource();
}

GPUResourceHandle SingleViewProcess::getColorCvtTexture()
{
	return m_2rgbColorCvt->getTargetGPUResource();
}

void SingleViewProcess::updateCameraParams()
{
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();
	CameraInput curCameraInput = gasettings->getCameraInput(camIndex);

	m_camWeightMap->setCameraInput(curCameraInput);
	m_unwarp->setCameraInput(curCameraInput);

	m_camWeightMap->updateCameraParams();
	m_unwarp->updateCameraParams();

	m_paramChanged = true;
}

/*void SingleViewProcess::processImgProc(ImageBufferData mat)
{
	frameMutex.lock();
	m_frame = mat;
	isFrameReady = true;
	frameMutex.unlock();
}

void SingleViewProcess::process()
{
	m_finished = false;
	run();
	thread()->terminate();
	finishWC.wakeAll();
	m_finished = true;
}

void SingleViewProcess::stop()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
}

void SingleViewProcess::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex);
	finishMutex.unlock();
}

bool SingleViewProcess::isFinished()
{
	return m_finished;
}*/

void SingleViewProcess::uploadTexture(ImageBufferData mat)
{
	//sharedImageBuffer->lockIncomingBuffer(i);
	m_2rgbColorCvt->render(mat);
	//sharedImageBuffer->unlockIncomingBuffer(i);
}

void SingleViewProcess::render()
{
	CameraInput camInput = sharedImageBuffer->getGlobalAnimSettings()->cameraSettingsList()[camIndex];

	m_gainCompensation->render(m_2rgbColorCvt->getTargetGPUResource(), camInput.exposure);

	if (m_paramChanged)
	{
		m_paramChanged = false;
		m_camWeightMap->render(camIndex);
	}

	m_unwarp->render(m_gainCompensation->getTargetGPUResource(), GLSLUnwarp::Color);
}

void SingleViewProcess::downloadTexture(unsigned char* liveGrabBuffer)
{
	m_gainCompensation->getRGBBuffer(liveGrabBuffer);
}

/*void SingleViewProcess::run()
{
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();
	GlobalAnimSettings::CameraSettingsList& camsettings = gasettings->cameraSettingsList();

	isFrameReady = false;

	int intervalms = 1;

	while (1)
	{
		if (QThread::currentThread()->isInterruptionRequested())
		{
			std::cout << "Got signal to terminate" << std::endl;
			doStop = true;
		}
		//
		// Stop thread if doStop = TRUE 
		//
		doStopMutex.lock();
		if (doStop)
		{
			//std::cout << "Stop" << std::endl;
			doStop = false;
			doStopMutex.unlock();
			break;
		}
		doStopMutex.unlock();

		bool isProcess;
		frameMutex.lock();
		isProcess = isFrameReady;
		isFrameReady = false;
		frameMutex.unlock();

		if (isProcess)
		{
			if (m_context != QOpenGLContext::currentContext())
				m_context->makeCurrent(m_surface);

			sharedImageBuffer->lockIncomingBuffer(camIndex);

			m_2rgbColorCvt->render(m_frame);

			sharedImageBuffer->unlockIncomingBuffer(camIndex);

			m_gainCompensation->render(m_2rgbColorCvt->getTargetGPUResource(), camsettings[camIndex].exposure);
			if (m_paramChanged)
			{
				m_paramChanged = false;
				m_camWeightMap->render(camIndex);
			}

			m_unwarp->render(m_gainCompensation->getTargetGPUResource(), m_lightColor, GLSLUnwarp::Color);

			m_context->doneCurrent();

			std::shared_ptr< D360Stitcher> stitcher = sharedImageBuffer->getStitcher();
			if (stitcher)
			{
				//stitcher->updateStitchFrame(camIndex);
			}
		}
		
		Sleep(intervalms);
	}
}

void SingleViewProcess::started()
{
	m_context->doneCurrent();
	m_context->moveToThread(thread());
}*/