#include <iostream>
#include "D360Stitcher.h"
#include <QQmlApplicationEngine>
#include "QmlMainWindow.h"
#include <QThread>
#include <QImage>
#include <QtGlobal>
#include <QOpenGL.h>
#include <stdio.h>  
#include <iostream>
#include <fstream>
#include <string>
#include <QtGui>
#include <QOffscreenSurface>
#include <QMutex>

#include "Buffer.h"
#include "define.h"
#include "Structures.h"
#include "ConfigZip.h"
#include "include/Config.h"

#ifdef USE_CUDA
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

extern QmlMainWindow* g_mainWindow;

#define ENABLE_LOG 1
#ifdef USE_CUDA
#define CUDA_LOG
#endif

using namespace std;

#define CUR_TIME							QTime::currentTime().toString("mm:ss.zzz")
#define ARGN(num)							arg(QString::number(num))

#ifdef _DEBUG
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
#endif  // _DEBUG

extern QThread* g_mainThread;
D360Stitcher* g_pStitcher = NULL;

#ifdef USE_CUDA
cudaStream_t g_CurStream = NULL;
cudaStream_t g_NextStream = NULL;
#endif

#ifdef USE_CUDA
DoStitchThread::DoStitchThread()
{
	m_bIsRunning = false;
	m_pContext = NULL;
};

DoStitchThread::~DoStitchThread()
{
	if (m_pContext)
	{
		delete m_pContext;
		m_pContext = NULL;
	}
}

void DoStitchThread::init()
{
	m_pContext = new QOpenGLContext();
	QOffscreenSurface* pSurface = g_pStitcher->getSurface();
	QSurfaceFormat format = pSurface->requestedFormat();
	format.setSwapInterval(0);
	format.setSwapBehavior(QSurfaceFormat::SingleBuffer);
	format.setVersion(4, 3);
	m_pContext->setFormat(format);
	m_pContext->setShareContext(QOpenGLContext::globalShareContext());
	m_pContext->create();

	m_pContext->moveToThread(this);
}

void DoStitchThread::run()
{
	QMutexLocker locker(&mutex);

	m_bIsRunning = true;

	QOffscreenSurface* pSurface = g_pStitcher->getSurface();
	if (m_pContext != QOpenGLContext::currentContext()){
		m_pContext->makeCurrent(pSurface);
	}	

	GlobalAnimSettings *pGaSettings = g_pStitcher->getGaSettings();
	g_pStitcher->DoCompositePanorama(pGaSettings);
	m_pContext->doneCurrent();

	m_bIsRunning = false;
}
#endif

//////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////// 

D360Stitcher::D360Stitcher(SharedImageBuffer *pSharedImageBuffer, QObject* main) : sharedImageBuffer(pSharedImageBuffer)
, m_stitcherThreadInstance(0)
, m_Main(main)
, m_Panorama(NULL)
, m_finalPanorama(NULL)
, m_nViewCount(0)
, m_fboBuffer(0)
, m_surface(0)
, m_context(0)
, cameraFrameProcN(0)
, m_Name("Stitcher")
, doSnapshot(false)
, m_gaSettings(NULL)
, srcWeightTextures(NULL)
, m_cudaThreadCount(0)
, m_idxCurrentGPU(0)
{
	connect(sharedImageBuffer, SIGNAL(fireEventWeightMapEditEnabled(bool)),
		this, SLOT(setWeightMapEditMode(bool)), Qt::DirectConnection);
	
	initialize();	

	m_finished = true;
	g_pStitcher = this;

	m_viewProcessors = NULL;
}


D360Stitcher::~D360Stitcher(void)
{
	disconnect(sharedImageBuffer, SIGNAL(fireEventWeightMapEditEnabled(bool)),
		this, SLOT(setWeightMapEditMode(bool)));

	qquit();

	reset();

	removeOutputNode();
}

void D360Stitcher::initialize()
{
	cameraFrameProcN = 0;
	doStop = false;
	sampleNumber = 0;
	fpsSum = 0;

	fps.clear();
	statsData.averageFPS = 0;
	statsData.nFramesProcessed = 0;
	statsData.elapesedTime = 0;

	doPause = false;
	doResetGain = false;
	doCalcGain = false;
	doReStitch = false;
	doUpdateCameraParams = false;

	removeBannerAll = false;
	removeBannerLast = false;
	removeBannerIndex = -1;

	m_weightMapChanged = false;
	m_weightMapCameraIndex = 0;
	m_isWeightMapReset = false;
	m_weightMapEyeMode = WeightMapEyeMode::DEFAULT;

	m_panoWeightRecordOn = false;	

	clear();
}

void D360Stitcher::initForReplay()
{
	sharedImageBuffer->initializeForReplay();
	initialize();
}

void D360Stitcher::init(QOpenGLContext* context)
{
	initialize();

	reset();

#ifdef USE_CUDA
	cudaGetDeviceCount(&m_cudaThreadCount);
	if (m_cudaThreadCount <= 1) {
		m_cudaThreadCount = 1;
	}

	for (int i = 0; i < m_cudaThreadCount; i++)
	{
		cudaSetDevice(i);
		cudaDeviceReset();

		size_t fm, tm;
		cudaError err = cudaMemGetInfo(&fm, &tm);
		if (err != cudaSuccess) {
			PANO_LOG(QString("Cuda %1 Error is occured. Total Memory : %2(MB), Usable Memory : %3(MB)").ARGN(i + 1).ARGN(tm / 1024 / 1024).ARGN(fm / 1024 / 1024));
		}
		else{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, i);
			//deviceProp.name
			PANO_LOG(QString("Device %1 [%4] Total Memory : %2(MB), Usable Memory : %3(MB)").ARGN(i + 1).ARGN(tm / 1024 / 1024).ARGN(fm / 1024 / 1024).arg(deviceProp.name));
		}
	}

	if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/)
	{
		saveTexture = saveTextureCUDA;
		m_bRunningByCuda = true;
	}
	else
#endif
	{
		m_cudaThreadCount = 1;
		m_bRunningByCuda = false;
		saveTexture = saveTextureGL;
	}

	//Comment below code for multi-gpu.
	m_cudaThreadCount = 1;

	m_gaSettings = sharedImageBuffer->getGlobalAnimSettings();

	m_surface = new QOffscreenSurface();
	m_surface->create();

	m_context = new QOpenGLContext();
	QSurfaceFormat format = m_surface->requestedFormat();
	format.setSwapInterval(0);
	format.setSwapBehavior(QSurfaceFormat::SingleBuffer);
	format.setVersion(4, 3);
	format.setProfile(QSurfaceFormat::CompatibilityProfile);
	m_context->setFormat(format);
	m_context->setShareContext(context);
	m_context->create();

	m_context->makeCurrent(m_surface);

	QOpenGLVersionProfile profile;
	profile.setProfile(m_context->surface()->format().profile());
	QOpenGLFunctions* gl = (QOpenGLFunctions*)m_context->functions();
	functions_2_0 = m_context->versionFunctions<QOpenGLFunctions_2_0>();
	functions_4_3 = m_context->versionFunctions<QOpenGLFunctions_4_3_Compatibility>();

	m_nPanoramaReadyCount = 0;

	int panoCount = m_gaSettings->isStereo() ? 2 : 1;
	int panoWidth = m_gaSettings->m_panoXRes;
	int panoHeight = m_gaSettings->m_panoYRes;

	m_nViewCount = m_gaSettings->cameraSettingsList().size();

	m_viewProcessors = new vector<SingleViewProcess *>[m_cudaThreadCount];

	for (int idx = 0; idx < m_cudaThreadCount; idx++)
	{
#ifdef USE_CUDA
		if (m_bRunningByCuda)
		{
			cudaSetDevice(idx);
		}
#endif
		for (int i = 0; i < m_nViewCount; i++)
		{
#ifdef CUDA_LOG
			size_t fm, tm, fm1;
			cudaMemGetInfo(&fm, &tm);
#endif
			int camWidth = m_gaSettings->getCameraInput(i).xres;
			int camHeight = m_gaSettings->getCameraInput(i).yres;

			SingleViewProcess *viewProcessor = new SingleViewProcess(sharedImageBuffer, i, camWidth, camHeight);
			viewProcessor->create(gl, functions_2_0, functions_4_3);
			m_viewProcessors[idx].push_back(viewProcessor);
#ifdef CUDA_LOG
			cudaMemGetInfo(&fm1, &tm);
			PANO_LOG(QString("Device %1, Camera %2 SingleViewProcess's Memory : %3(MB)").ARGN(idx + 1).ARGN(i + 1).ARGN((fm - fm1) / 1024 / 1024));
#endif
		}
	}

	if (m_gaSettings->isNodalAvailable())
	{
		// for Multi-Nodal
		for (unsigned i = 0; i < m_gaSettings->m_nodalVideoFilePathMap.keys().size(); i ++)
		{
			m_nodalColorCvtMap[i] = new GPUNodalInput*[m_cudaThreadCount];
			for (int idx = 0; idx < m_cudaThreadCount; idx++){

#ifdef USE_CUDA
				if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
 					cudaSetDevice(idx);
 					//m_nodalColorCvt[idx] = new CUDANodalInput();
				}
				else
#endif
				{
					m_nodalColorCvtMap[i][idx] = new GLSLNodalInput();
				}

				int slotIndex = m_gaSettings->m_nodalMaskImageFilePathMap.keys()[i];
				m_nodalColorCvtMap[i][idx]->createColorCvt(gl, m_gaSettings->isLiveNodal(), m_gaSettings->getNodalWeightPath(slotIndex));
			}
		}
	}

	int imgWidth = m_gaSettings->getCameraInput(0).xres;
	int imgHeight = m_gaSettings->getCameraInput(0).yres;
	
	liveGrabBuffer = new unsigned char[imgWidth * imgHeight * 3];

	m_panoramaUnits = new vector<SinglePanoramaUnit *>[m_cudaThreadCount];
	for (int idx = 0; idx < m_cudaThreadCount; idx++)
	{
#ifdef USE_CUDA
		if (m_bRunningByCuda) {
			cudaSetDevice(idx);
		}
#endif
		for (int i = 0; i < panoCount; i++)
		{
			SinglePanoramaUnit * panoramaUnit = new SinglePanoramaUnit(sharedImageBuffer);
			panoramaUnit->setGL(gl, functions_2_0, functions_4_3);
			m_panoramaUnits[idx].push_back(panoramaUnit);
		}
	}

	//For WeightMap Editing
	GLSLDeltaWeightMap::vertices = NULL;
	m_weightMapEditPoints.clear();

	m_Panorama = new GPUPanorama*[m_cudaThreadCount];
	m_finalPanorama = new GPUFinalPanorama*[m_cudaThreadCount];

	for (int idx = 0; idx < m_cudaThreadCount; idx++)
	{
#ifdef USE_CUDA
		if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/)
		{
			cudaSetDevice(idx);
			m_Panorama[idx] = new CUDAPanorama();
			m_finalPanorama[idx] = new CUDAFinalPanorama();
		}
		else
#endif
		{
			m_Panorama[idx] = new GLSLPanorama();
			m_finalPanorama[idx] = new GLSLFinalPanorama();
		}


		m_Panorama[idx]->setGL(gl, functions_2_0);
		m_finalPanorama[idx]->setGL(gl, functions_2_0);
	}

	for (unsigned i = 0; i < m_nodalColorCvtMap.keys().size(); i++)
	{
		for (int idx = 0; idx < m_cudaThreadCount; idx++)
		{
#ifdef USE_CUDA
			if (m_bRunningByCuda) {
				cudaSetDevice(idx);
			}
#endif
#ifdef CUDA_LOG
			size_t fm, tm, fm1;
			cudaMemGetInfo(&fm, &tm);
#endif
				int camWidth = m_gaSettings->getCameraInput(NODAL_CAMERA_INDEX + i).xres;
				int camHeight = m_gaSettings->getCameraInput(NODAL_CAMERA_INDEX + i).yres;
				m_nodalColorCvtMap[i][idx]->initialize(camWidth, camHeight);
#ifdef CUDA_LOG
			cudaMemGetInfo(&fm1, &tm);
			PANO_LOG(QString("Device %1, NodalInput's Memory : %2(MB)").ARGN(idx + 1).ARGN((fm - fm1) / 1024 / 1024));
#endif
		}
	}

	m_isUndoRedo = false;
	m_undoRedoBufferIdx = 0;
	for (int i = 0; i < m_undoRedo.size(); i++) {
		GLuint texId = m_undoRedo[i].leftUndoRedoTexId;
		m_context->functions()->glDeleteTextures(1, &texId);
		texId = m_undoRedo[i].rightUndoRedoTexId;
		m_context->functions()->glDeleteTextures(1, &texId);
	}
	m_undoRedo.clear();
	m_undoRedoState = NONE_UNDOREDO;
	m_weightmapCamIdxChangedByUndoRedo = false;

	m_weightMapEditPoints.clear();
	m_colorTemperature = 6600;
	m_lightColor = vec3(1, 1, 1);

	// create texture id list for left and right unwarped images in panorama space
	int viewCnt[2] = { m_nViewCount, 0 };
	GPUResourceHandle **colorFboTextures = new GPUResourceHandle *[m_cudaThreadCount];
	srcWeightTextures = new std::vector<GPUResourceHandle>[m_cudaThreadCount];

	for (int idx = 0; idx < m_cudaThreadCount; idx++)
	{
		colorFboTextures[idx] = new GPUResourceHandle[8];
#ifdef USE_CUDA
		if (m_bRunningByCuda) {
			cudaSetDevice(idx);
		}
#endif
		for (int i = 0; i < m_nViewCount; i++)
		{
			colorFboTextures[idx][i] = m_viewProcessors[idx][i]->getUnwarpTexture();
			srcWeightTextures[idx].push_back(m_viewProcessors[idx][i]->getWeightTexture());
		}
	}

	if (m_gaSettings->isStereo())
	{
		viewCnt[0] = m_gaSettings->getLeftIndices().size();
		viewCnt[1] = m_gaSettings->getRightIndices().size();
	}
	else
	{
		m_gaSettings->getLeftIndices().clear();
		for (int i = 0; i < m_nViewCount; i++)
		{
			m_gaSettings->getLeftIndices().push_back(i);
		}
	}
	
	for (int idx = 0; idx < m_cudaThreadCount; idx++)
	{
#ifdef USE_CUDA
		if (m_bRunningByCuda) {
			cudaSetDevice(idx);
		}
#endif
		for (int i = 0; i < panoCount; i++)
		{
#ifdef CUDA_LOG
			size_t fm, tm, fm1;
			cudaMemGetInfo(&fm, &tm);
#endif
			m_panoramaUnits[idx][i]->initialize(panoWidth, panoHeight, viewCnt[i], i,
				colorFboTextures[idx], m_nodalColorCvtMap.keys().size(), m_gaSettings->m_haveNodalMaskImage);
#ifdef CUDA_LOG
			cudaMemGetInfo(&fm1, &tm);
			PANO_LOG(QString("Device %1, %2 Single Panorama Unit Memory : %3(MB)").ARGN(idx + 1).ARGN(i + 1).ARGN((fm - fm1) / 1024 / 1024));
#endif
		}
#ifdef CUDA_LOG
		size_t fm, tm, fm1;
		cudaMemGetInfo(&fm, &tm);
#endif
		m_Panorama[idx]->initialize(panoWidth, panoHeight, m_gaSettings->isStereo());
#ifdef CUDA_LOG
		cudaMemGetInfo(&fm1, &tm);
		PANO_LOG(QString("Device %1, Panorama's Memory : %2(MB)").ARGN(idx + 1).ARGN((fm - fm1) / 1024 / 1024));
		fm = fm1;
#endif
		m_finalPanorama[idx]->initialize(panoWidth, panoHeight, m_gaSettings->isStereo());
#ifdef CUDA_LOG
		cudaMemGetInfo(&fm1, &tm);
		PANO_LOG(QString("Device %1, Final Panorama's Memory : %2(MB)").ARGN(idx + 1).ARGN((fm - fm1) / 1024 / 1024));
		PANO_LOG(QString("Device %1, Remainder Memory : %2(MB)").ARGN(idx + 1).ARGN(fm1 / 1024 / 1024));
#endif
		delete[] colorFboTextures[idx];
	}
	delete[] colorFboTextures;

	// 
	m_context->doneCurrent();

	isFirstFrame = true;

#ifdef USE_CUDA
	m_Thread = new DoStitchThread[m_cudaThreadCount];
	for (int i = 0; i < m_cudaThreadCount; i++) {
		m_Thread[i].init();
	}

	g_CurStream = g_NextStream = NULL;

	for (int i = 0; i < 2; i++) {
		cudaStreamCreate(&m_stream[i]);
	}
#endif
}

void D360Stitcher::buildOutputNode(int width, int height)
{
	QOpenGLFunctions* gl = (QOpenGLFunctions*)m_context->functions();
	GlobalAnimSettings* gasettings = sharedImageBuffer->getGlobalAnimSettings();

	removeOutputNode();
	// Panorama resolution will be 4k * 2k,
	// for stereo mode, the top-down video resolution will be 4k * 4k, (which is w by 2h)
	// for mono mode, the resolution will be 4k * 2k (which is w by h)
	int panoramaFrameBytes = width * height * 3 / 2;
	outputBufferMutex.lock();	if (gasettings->isStereo())
	{
		m_fboBuffer = new unsigned char[panoramaFrameBytes * 2];
	}
	else
	{
		m_fboBuffer = new unsigned char[panoramaFrameBytes];
	}
	outputBufferMutex.unlock();
	for (int idx = 0; idx < m_cudaThreadCount; idx++)
	{
		m_finalPanorama[idx]->requestReconfig(width, height, gasettings->isStereo());
	}
}

void D360Stitcher::qquit()
{
	PANO_LOG("Stitcher Finished ");
	if (m_stitcherThreadInstance && m_stitcherThreadInstance->isRunning())
		stopStitcherThread();
	releaseStitcherThread();
}

void D360Stitcher::releaseStitcherThread()
{
	if (m_stitcherThreadInstance)
	{
		m_stitcherThreadInstance->quit();
		m_stitcherThreadInstance->wait();
		delete m_stitcherThreadInstance;
		m_stitcherThreadInstance = NULL;
	}
}

void D360Stitcher::stopStitcherThread()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
	sharedImageBuffer->wakeStitcher();
}

void D360Stitcher::setup()
{
	releaseStitcherThread();
	
	m_stitcherThreadInstance = new QThread;
	this->moveToThread(m_stitcherThreadInstance);
	connect(m_stitcherThreadInstance, SIGNAL(started()), this, SLOT(process()));
	//connect(processingThread, SIGNAL(finished()), processingThreadInstance, SLOT(quit()));
	//connect(m_stitcherThreadInstance, SIGNAL(finished()), this, SLOT(deleteLater()));
	//connect(this, SIGNAL(finished(int, QString, int)), m_stitcherThreadInstance, SLOT(deleteLater()));
}

void D360Stitcher::startThread()
{
	setup();
	m_stitcherThreadInstance->start();
	m_context->doneCurrent();
	m_context->moveToThread(m_stitcherThreadInstance);
}

QOpenGLContext* D360Stitcher::getContext()
{
	return m_context;
}

QOffscreenSurface* D360Stitcher::getSurface()
{
	return m_surface;
}

GlobalAnimSettings* D360Stitcher::getGaSettings()
{
	return m_gaSettings;
}

void D360Stitcher::setLutData(QVariantList *vList)
{
	if (m_idxCurrentGPU < 0 || m_idxCurrentGPU >= m_panoramaUnits->size()) return;
	for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++){
		m_panoramaUnits[m_idxCurrentGPU][i]->setLutData(vList);
	}
}

int D360Stitcher::getRawTextureId(int cameraIndex)
{
	if (m_idxCurrentGPU < 0 || m_idxCurrentGPU >= m_viewProcessors->size()) return -1;
	if (m_viewProcessors[m_idxCurrentGPU].size() <= cameraIndex || cameraIndex < 0) return -1;
	return m_viewProcessors[m_idxCurrentGPU][cameraIndex]->getRawTexture();
}

int D360Stitcher::getPanoramaTextureId()
{
	if (m_idxCurrentGPU < 0 || m_Panorama == NULL) return -1;
	if (m_Panorama[m_idxCurrentGPU] == NULL) return -1;
	return m_Panorama[m_idxCurrentGPU]->getTargetGPUResourceForInteract();
}

GPUResourceHandle D360Stitcher::getPanoramaFBO()
{
	if (m_idxCurrentGPU < 0 || m_Panorama == NULL) return -1;
	if (m_Panorama[m_idxCurrentGPU] == NULL) return -1;
	return m_Panorama[m_idxCurrentGPU]->getTargetBuffer();
}

int D360Stitcher::getPanoramaTextureIdForInteract()
{
	if (m_idxCurrentGPU < 0 || m_idxCurrentGPU >= m_panoramaUnits->size()) return -1;
	if (m_panoramaUnits[m_idxCurrentGPU].size() > 0)
		return m_panoramaUnits[m_idxCurrentGPU][0]->getPanoramaTexture();
	return -1;
}

void D360Stitcher::updateStitchFrame(ImageBufferData& frame, int camIndex)
{
	QMutexLocker locker(&m_stitchMutex);

	cameraFrameRaw[camIndex] = frame;
	if (cameraFrameRaw.size() == sharedImageBuffer->getSyncedCameraCount())
	{
		cameraFrameUse = cameraFrameRaw;
		cameraFrameProcN++;

		cameraFrameRaw.clear();
		sharedImageBuffer->wakeStitcher();
	}
}

void D360Stitcher::lockBannerMutex()
{
	m_bannerMutex.lock();
}

void D360Stitcher::unlockBannerMutex()
{
	m_bannerMutex.unlock();
}

BannerInfo& D360Stitcher::createNewBanner()
{
	BannerInfo banner;
	banner.id = BannerInfo::seedId++;
	bannerInfos.push_back(banner);
	return bannerInfos[bannerInfos.size() - 1];
}

BannerInfo* D360Stitcher::getBannerLast()
{
	int last = bannerInfos.size() - 1;
	if (last < 0)
		return NULL;
	BannerInfo* banner = &bannerInfos[last];
	return banner;
}

BannerInfo* D360Stitcher::getBannerAtIndex(int index)
{	
	BannerInfo* banner = &bannerInfos[index];
	return banner;
}

void D360Stitcher::cueRemoveBannerAll()
{
	removeBannerAll = true;
}

void D360Stitcher::cueRemoveBannerLast()
{
	removeBannerLast = true;
}

void D360Stitcher::cueRemoveBannerAtIndex(int index)
{
	removeBannerIndex = index;
}

void D360Stitcher::updateBannerVideoFrame(ImageBufferData& frame, int bannerId)
{
	for (int i = 0; i < bannerInfos.size(); i++)
	{
		BannerInfo& banner = bannerInfos[i];
		if (banner.id == bannerId)
		{
			banner.frame = frame;
			break;
		}
	}
}

void D360Stitcher::updateBannerImageFrame(QImage image, int bannerId)
{
	QImage frameImage = image.convertToFormat(QImage::Format_RGBA8888);
	for (int i = 0; i < bannerInfos.size(); i++)
	{
		BannerInfo& banner = bannerInfos[i];
		if (banner.id == bannerId)
		{
			ImageBufferData frame(ImageBufferData::RGBA8888);
			frame.mImageY.makeBuffer(frameImage.height() * frameImage.bytesPerLine());
			frame.mImageY.setImageAttribute(frameImage);
			memcpy(frame.mImageY.buffer, frameImage.constBits(), frameImage.height() * frameImage.bytesPerLine());
			banner.frame = frame;
			break;
		}
	}
}

void D360Stitcher::saveBanner(GlobalAnimSettings& settings)
{
	settings.m_banners.clear();

	for (int i = 0; i < bannerInfos.size(); i++)
	{
		BannerInfo& banner = bannerInfos[i];
		
		BannerInput info;
		info.bannerFile = banner.filePath;
		info.isStereoRight = banner.isStereoRight;
		info.isVideo = banner.isVideo;
		memcpy(info.quad, banner.quad, sizeof(banner.quad));

		settings.m_banners.push_back(info);
	}
}

void D360Stitcher::saveWeightMaps(QString iniPath)
{
	if (!m_gaSettings)
		return;

	QString strWeightMap = "WeightMap";
	QString strSaveDir = iniPath.left(iniPath.lastIndexOf("/"));

	QMutexLocker locker(&doWeightMapSaveMutex);

//	if (m_gaSettings->m_weightMaps.size() == 0) {
	m_gaSettings->m_weightMaps.clear();
	int panoCount = m_gaSettings->isStereo() ? 2 : 1;
	for (int k = 0; k < panoCount; k++)
	{
		for (int cnt = 0; cnt < m_gaSettings->getCameraCount(); cnt++)
		{
			QString imgName = QString(strSaveDir + "/weightMap%1_%2.png").arg(k).arg(cnt);

			WeightMapInput info;
			info.weightMapFile = imgName;
			m_gaSettings->m_weightMaps.push_back(info);
		}
	}
//	}
	m_gaSettings->m_weightMapDir = iniPath;
	
	QOpenGLContext *pContext = new QOpenGLContext();
	QOffscreenSurface* pSurface = getSurface();
	QSurfaceFormat format = pSurface->requestedFormat();
	format.setSwapInterval(0);
	format.setSwapBehavior(QSurfaceFormat::SingleBuffer);
	format.setVersion(4, 3);
	pContext->setFormat(format);
	pContext->setShareContext(QOpenGLContext::globalShareContext());
	pContext->create();

	pContext->makeCurrent(pSurface);
	processSaveWeightMaps();
	pContext->doneCurrent();

	delete pContext;
}

void D360Stitcher::recordMaskMap(QString maskPath)
{
	m_stitchMutex.lock();
	m_panoWeightPath = maskPath;
	m_panoWeightRecordOn = true;
	m_panoWeightSkipFrames = 0;
	m_stitchMutex.unlock();
}

void D360Stitcher::processSaveWeightMaps()
{	
	int panoCount = m_gaSettings->isStereo() ? 2 : 1;
	for (int k = 0; k < panoCount; k++)
	{
		m_panoramaUnits[m_idxCurrentGPU][k]->saveWeightMaps();
	}
}

bool D360Stitcher::isCameraReadyToProcess(int camIndex)
{
	if (sharedImageBuffer->getLiveGrabber() == -1)
		return true;
	else
	{
		return camIndex == sharedImageBuffer->getLiveGrabber();
	}
}

void D360Stitcher::doCaptureIncomingFrames()
{
	//PANO_LOG("doCaptureIncomingFrames - 1");
	m_stitchMutex.lock();
	map<int, ImageBufferData> frames = cameraFrameUse;
	m_stitchMutex.unlock();


// 	LARGE_INTEGER st, et, fq;
// 	::QueryPerformanceCounter(&st);
// 	::QueryPerformanceFrequency(&fq);
	//PANO_LOG_ARG("doCaptureIncomingFrames - 2 [%1]", m_2rgbColorCvt.size());
	for (map<int, ImageBufferData>::iterator iter = frames.begin(); iter != frames.end(); iter++)
	{
		int i = iter->first;

		//if (!isCameraReadyToProcess(i))
		//	continue;

		if (i >= 0)
		{
			m_viewProcessors[m_idxCurrentGPU][i]->uploadTexture(iter->second);
		}
		// for Multi-Nodal
		else if (IS_NODAL_CAMERA_INDEX(i))
		{
			for (unsigned i = 0; i < m_nodalColorCvtMap.keys().size(); i ++)
			{
				m_nodalColorCvtMap[i][m_idxCurrentGPU]->render(iter->second);
			}
		}
	}
// 	::QueryPerformanceCounter(&et);
// 	qDebug().noquote() << QString("texture uploading time is %1(ms)\n").arg((et.QuadPart - st.QuadPart) / (double)fq.QuadPart * 1000.f);
	//PANO_LOG("doCaptureIncomingFrames - 4");
}

void D360Stitcher::processIndividualViews(bool weightMapChangedByCameraParam)
{
// 	LARGE_INTEGER st, et, fq;
// 	::QueryPerformanceCounter(&st);
// 	::QueryPerformanceFrequency(&fq);
	GlobalAnimSettings& setting = g_mainWindow->getGlobalAnimSetting();
	GlobalAnimSettings::CameraSettingsList& camsettings = setting.cameraSettingsList();

	for (int i = 0; i < m_nViewCount; i++)
	{
		//if (!isCameraReadyToProcess(i)) continue;

		m_viewProcessors[m_idxCurrentGPU][i]->render();

		if (sharedImageBuffer->getLiveGrabber() != -1 && sharedImageBuffer->getLiveGrabber() == i)
		{
			m_viewProcessors[m_idxCurrentGPU][i]->downloadTexture(liveGrabBuffer);
			g_mainWindow->newCalibFrame(liveGrabBuffer, camsettings[i].xres, camsettings[i].yres);
		}
	}
// 	::QueryPerformanceCounter(&et);
// 	qDebug().noquote() << QString("individual processing time is %1(ms)\n").arg((et.QuadPart - st.QuadPart) / (double)fq.QuadPart * 1000.f);
}

void D360Stitcher::updateWeightmap(bool weightMapChanged)
{
	if (m_isWeightMapReset)
	{
		resetEditedWeightMap();
	}
	else if (sharedImageBuffer->isWeightMapEditEnabled())
	{
		makeCameraWeightMap();
	}

	if (!m_isWeightMapReset && sharedImageBuffer->isWeightMapEditEnabled())
	{
		if (m_weightMapEyeMode == WeightMapEyeMode::BOTHMODE){
			for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++){
				m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, m_weightMapCameraIndex, srcWeightTextures[m_idxCurrentGPU]);
			}
		}
		else if (m_weightMapEyeMode == WeightMapEyeMode::MIRROR)
		{
			for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++) {
				if (i == 0) {
					m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, m_weightMapCameraIndex, srcWeightTextures[m_idxCurrentGPU]);
				}
				else if (i == 1) {
					m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, m_weightMapCameraIndex_, srcWeightTextures[m_idxCurrentGPU]);
				}
			}
		}
		else{
			int idx = (m_weightMapEyeMode == WeightMapEyeMode::RIGHTMODE) ? 1 : 0;
			for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++){
				if (i == idx)
					m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, m_weightMapCameraIndex, srcWeightTextures[m_idxCurrentGPU]);
				else
					m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, -1, srcWeightTextures[m_idxCurrentGPU]);
 			}
		}
	}
	else if (weightMapChanged){
		for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++)
		{
			m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, -1, srcWeightTextures[m_idxCurrentGPU]);
		}
	}
	else{
		for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++)
		{
			m_panoramaUnits[m_idxCurrentGPU][i]->updateWeightMap(weightMapChanged, -2, srcWeightTextures[m_idxCurrentGPU]);
		}
	}
}

void D360Stitcher::stitchPanorama()
{
	m_bannerMutex.lock();
	bool hasBanner = !bannerInfos.empty();
	if (hasBanner)
	{
		for (int i = 0; i < bannerInfos.size(); i++)
		{
			BannerInfo& banner = bannerInfos[i];
			banner.isValid = banner.billColorCvt->DynRender(banner.frame);
		}
	}
	m_bannerMutex.unlock();

	int nodalTexture = -1;
	int nodalWeightTexture = -1;
	QList<int> nodalTextures; nodalTextures.clear();
	QList<int> nodalWeightTextures; nodalWeightTextures.clear();

	for (int i = 0; i < m_nodalColorCvtMap.keys().size(); i++) {
			nodalTexture = m_nodalColorCvtMap[i][m_idxCurrentGPU]->getColorCvtGPUResource();
			nodalWeightTexture = m_nodalColorCvtMap[i][m_idxCurrentGPU]->getWeightGPUResource();
			nodalTextures.append(nodalTexture);
			nodalWeightTextures.append(nodalWeightTexture);
	}
// 	LARGE_INTEGER st, et, fq;
// 	::QueryPerformanceCounter(&st);
// 	::QueryPerformanceFrequency(&fq);
	m_bannerMutex.lock();
	for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++)
	{
		if (false/*g_mainWindow->m_eyeMode == 4*/)
		{
			if (i == 0) //Left
				m_panoramaUnits[m_idxCurrentGPU][i]->render(nodalTextures, nodalWeightTextures, bannerInfos,
				m_WeightmapPaintingMode, getAvailableIndex(i, m_weightMapCameraIndex), m_weightMapEyeMode, m_lightColor);
			if (i == 1) //Right
				m_panoramaUnits[m_idxCurrentGPU][i]->render(nodalTextures, nodalWeightTextures, bannerInfos,
				m_WeightmapPaintingMode, getAvailableIndex(i, m_weightMapCameraIndex_), m_weightMapEyeMode, m_lightColor);
		}

		if (!(false/*g_mainWindow->m_eyeMode == 4*/)) // MIRROR
			m_panoramaUnits[m_idxCurrentGPU][i]->render(nodalTextures, nodalWeightTextures, bannerInfos,
				m_WeightmapPaintingMode, getAvailableIndex(i, m_weightMapCameraIndex), m_weightMapEyeMode, m_lightColor);
	}
	m_bannerMutex.unlock();
// 	::QueryPerformanceCounter(&et);
// 	qDebug().noquote() << QString("m_panoramaUnits render time is %1(ms)\n").arg((et.QuadPart - st.QuadPart) / (double)fq.QuadPart * 1000.f);

	GPUResourceHandle individualPanoramaTextures[2];
	for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++)
		individualPanoramaTextures[i] = m_panoramaUnits[m_idxCurrentGPU][i]->getPanoramaTexture();

	m_Panorama[m_idxCurrentGPU]->render(individualPanoramaTextures, true);
	m_finalPanorama[m_idxCurrentGPU]->render(m_Panorama[m_idxCurrentGPU]->getTargetGPUResource());
}

vec3 D360Stitcher::calculateLightColorWithTemperature(float inTemp)
{
	vec3 lightColor;
	double inTemperature = (double)inTemp / 100.0;

	//Red
	if (inTemperature <= 66.0)
		lightColor.x = 1.0;
	else
	{
		lightColor.x = inTemperature - 60.0;
		lightColor.x = 329.698727446 * pow(lightColor.x, -0.1332047592);
		lightColor.x /= 255.0;
		if (lightColor.x < 0.0)
			lightColor.x = 0.0;
		else if (lightColor.x > 1.0)
			lightColor.x = 1.0;
	}

	//Green
	if (inTemperature <= 66.0)
	{
		lightColor.y = inTemperature;
		lightColor.y = 99.4708025861 * log(lightColor.y) - 161.1195681661;
		lightColor.y /= 255.0;
		if (lightColor.y < 0.0)
			lightColor.y = 0.0;
		else if (lightColor.y > 1.0)
			lightColor.y = 1.0;
	}
	else
	{
		lightColor.y = inTemperature - 60.0;
		lightColor.y = 288.1221695283 * pow(lightColor.y, -0.0755148492);
		lightColor.y /= 255.0;
		if (lightColor.y < 0.0)
			lightColor.y = 0.0;
		else if (lightColor.y > 1.0)
			lightColor.y = 1.0;
	}

	//Blue
	if (inTemperature >= 66.0)
		lightColor.z = 1.0;
	else if (inTemperature <= 19)
		lightColor.z = 0.0;
	else
	{
		lightColor.z = inTemperature - 10;
		lightColor.z = 138.5177312231 * log(lightColor.z) - 305.0447927307;
		lightColor.z /= 255.0;

		if (lightColor.z < 0.0)
			lightColor.z = 0.0;
		else if (lightColor.z > 1.0)
			lightColor.z = 1.0;
	}
	m_lightColor = lightColor;
	return lightColor;
}

void D360Stitcher::doStitch(bool stitchOn)
{
	m_bannerMutex.lock();

	if (removeBannerAll)
	{
		for (int i = 0; i < bannerInfos.size(); i++)
			bannerInfos[i].dispose();
		bannerInfos.clear();
		removeBannerAll = false;
	}
	if (removeBannerLast)
	{
		int last = bannerInfos.size() - 1;
		bannerInfos[last].dispose();
		bannerInfos.pop_back();
		removeBannerLast = false;
	}
	if (removeBannerIndex >= 0)
	{		
		bannerInfos[removeBannerIndex].dispose();
		bannerInfos.erase(bannerInfos.begin() + removeBannerIndex);
		removeBannerIndex = -1;
	}
	m_bannerMutex.unlock();

	processIndividualViews(m_isCameraParamChanged);
	if (stitchOn)
		doMakePanorama();
}

void D360Stitcher::doOutput()
{
// 	LARGE_INTEGER st, et, fq;
// 	::QueryPerformanceCounter(&st);
// 	::QueryPerformanceFrequency(&fq);

	// Only export to HDD or RTMP when current frame is new.
	// (Do not want to export the redundant frames
	if (!sharedImageBuffer->getStreamer()) return;

	// Save pano weight
	QString panoWeightPath;
	m_stitchMutex.lock();
	if (m_panoWeightRecordOn)
	{
		m_panoWeightSkipFrames++;
		if (m_panoWeightSkipFrames > 2)
		{
			m_panoWeightRecordOn = false;
			panoWeightPath = m_panoWeightPath;
		}
	}
	m_stitchMutex.unlock();
	if (panoWeightPath != "")
	{
		int panoWidth = sharedImageBuffer->getGlobalAnimSettings()->m_panoXRes;
		int panoHeight = sharedImageBuffer->getGlobalAnimSettings()->m_panoYRes;
		if (m_gaSettings->isStereo())
			panoHeight *= 2;
		unsigned char* alphaBuffer = new unsigned char[panoWidth * panoHeight * 2];
		m_Panorama[m_idxCurrentGPU]->downloadTexture(alphaBuffer);
		QImage panoWeightImage(alphaBuffer, panoWidth, panoHeight, QImage::Format::Format_Grayscale8);
		panoWeightImage.save(panoWeightPath);
		delete[] alphaBuffer;
	}

	// stream buffer
	outputBufferMutex.lock();
	if (m_fboBuffer)
	{
		m_finalPanorama[m_idxCurrentGPU]->downloadTexture(m_fboBuffer);
		emit newPanoramaFrameReady(m_fboBuffer);
	}
	outputBufferMutex.unlock();
// 	::QueryPerformanceCounter(&et);
// 	qDebug().noquote() << QString("output time is %1(ms)\n").arg((et.QuadPart - st.QuadPart) / (double)fq.QuadPart * 1000.f);
}

void D360Stitcher::doMakePanorama()
{
//	LARGE_INTEGER st, et, fq;
//	::QueryPerformanceCounter(&st);
//	::QueryPerformanceFrequency(&fq);
	// Render boundary map and seam only when there's camera parameter or seam index update
	updateWeightmap(m_isCameraParamChanged || (sharedImageBuffer->isWeightMapEditEnabled() && m_weightMapChanged) || m_isWeightMapReset || m_isUndoRedo);
	m_isCameraParamChanged = false;
	m_isWeightMapReset = false;

	stitchPanorama();
//	::QueryPerformanceCounter(&et);
//	qDebug().noquote() << QString("panorama making time time is %1(ms)\n").arg((et.QuadPart - st.QuadPart) / (double)fq.QuadPart * 1000.f);
}


void D360Stitcher::calcExposure()
{
	GlobalAnimSettings::CameraSettingsList& camsettings = g_mainWindow->getGlobalAnimSetting().cameraSettingsList();

	GPUResourceHandle fbos[16];
	for (int i = 0; i < m_nViewCount; i++)
	{
		fbos[i] = m_viewProcessors[m_idxCurrentGPU][i]->getTargetFrameBuffer();
	}

	if (m_gaSettings->isStereo())
	{
		m_panoramaUnits[m_idxCurrentGPU][0]->calcExposure(fbos);
		m_panoramaUnits[m_idxCurrentGPU][1]->calcExposure(fbos);
	}
	else
	{
		m_panoramaUnits[m_idxCurrentGPU][0]->calcExposure(fbos);
	}
	doStitch();
}

void D360Stitcher::reset()
{
	if (m_surface)
	{
		if (m_context != QOpenGLContext::currentContext())
			m_context->makeCurrent(m_surface);
		for (int idx = 0; idx < m_cudaThreadCount; idx++)
		{
			for (int i = 0; i < m_nViewCount; i++)
				delete m_viewProcessors[idx][i];
			m_viewProcessors[idx].clear();

			if (m_nodalColorCvtMap.keys().size() != 0)
				m_nodalColorCvtMap.clear();

			for (int i = 0; i < m_nodalColorCvtMap.keys().size(); i++) 
			{
				GPUNodalInput** nodalColorCvt = m_nodalColorCvtMap.values()[i];
				delete nodalColorCvt[idx];
			}

			if (m_nodalColorCvtMap.values().size() != 0)
			{
				m_nodalColorCvtMap.clear();
			}
		}

		delete[] m_viewProcessors;

		if (liveGrabBuffer)
		{
			delete[] liveGrabBuffer;
			liveGrabBuffer = NULL;
		}

		for (int idx = 0; idx < m_cudaThreadCount; idx++)
		{
			for (int i = 0; i < m_panoramaUnits[idx].size(); i++)
				delete m_panoramaUnits[idx][i];
			m_panoramaUnits[idx].clear();

			if (m_Panorama[idx])
				delete m_Panorama[idx];
			if (m_finalPanorama[idx])
				delete m_finalPanorama[idx];
		}

		delete[] m_panoramaUnits;
		delete[] m_Panorama;
		delete[] m_finalPanorama;
		m_panoramaUnits = NULL;
		m_Panorama = NULL;
		m_finalPanorama = NULL;

		for (int idx = 0; idx < m_cudaThreadCount; idx++)
		{
			srcWeightTextures[idx].clear();
		}
		delete[] srcWeightTextures;
		
		for (int i = 0; i < bannerInfos.size(); i++)
			bannerInfos[i].dispose();
		bannerInfos.clear();

		if (GPUDeltaWeightMap::vertices)
		{
			delete[] GPUDeltaWeightMap::vertices;
			GPUDeltaWeightMap::vertices = NULL;
		}

		m_undoRedoBufferIdx = 0;
		for (int i = 0; i < m_undoRedo.size(); i++)
		{
			GLuint texId = m_undoRedo[i].leftUndoRedoTexId;
			m_context->functions()->glDeleteTextures(1, &texId);
			texId = m_undoRedo[i].rightUndoRedoTexId;
			m_context->functions()->glDeleteTextures(1, &texId);
		}
		m_undoRedo.clear();
		m_undoRedoState = NONE_UNDOREDO;

		m_weightMapEditPoints.clear();

#ifdef USE_CUDA
		delete[] m_Thread;
		for (int i = 0; i < 2; i++)
		{
			cudaStreamDestroy(m_stream[i]);
		}		
#endif

		m_surface->destroy();
		m_context->doneCurrent();
		delete m_context;
		m_context = NULL;
		delete m_surface;
		m_surface = NULL;
	}
}

void D360Stitcher::removeOutputNode()
{
	outputBufferMutex.lock();
	if (m_fboBuffer)
	{
		delete[] m_fboBuffer;
		m_fboBuffer = NULL;
	}
	outputBufferMutex.unlock();
}

void D360Stitcher::clear()
{
	m_nSeamViewId = -1;
	m_isCameraParamChanged = true;

	cameraFrameRaw.clear();
	cameraFrameUse.clear();
}

void D360Stitcher::process()
{
	m_finished = false;
	emit started(THREAD_TYPE_STITCHER, "", -1);
	run();
	PANO_LOG("About to stop stitcher thread...");
	finishWC.wakeAll();	

	if (this == NULL)
	{
		PANO_ERROR("Stitcher instance unavailable...");
	}

	sharedImageBuffer->wakeAll(); // This allows the thread to be stopped if it is in a wait-state	

	PANO_LOG("Clear stitcher instance...");
	clear();	
	PANO_LOG("Stitcher thread successfully stopped.");		

	emit finished(THREAD_TYPE_STITCHER, "", -1);
	PANO_LOG("Stitcher - Emit Finished");
	/*while (!this->isFinished())
		Sleep(100);*/
		
	this->moveToThread(g_mainThread);
	if (m_context)
		m_context->moveToThread(g_mainThread);
}

void D360Stitcher::copyFromUndoRedoBuffer(int panoUnitIdx, int weightmapCamIndex)
{
	GLenum glerr = glGetError();
	int w = m_gaSettings->getCameraInput(weightmapCamIndex).xres;
	int h = m_gaSettings->getCameraInput(weightmapCamIndex).yres;
	
	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, m_panoramaUnits[m_idxCurrentGPU][panoUnitIdx]->getDeltaWeightMapFrameBuffer(weightmapCamIndex));
	m_context->functions()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, ((panoUnitIdx == 0) ? m_undoRedo[m_undoRedoBufferIdx].leftUndoRedoTexId : m_undoRedo[m_undoRedoBufferIdx].rightUndoRedoTexId), 0);
	m_context->functions()->glBindTexture(GL_TEXTURE_2D, m_panoramaUnits[m_idxCurrentGPU][panoUnitIdx]->getDeltaWeightTexture(weightmapCamIndex));
	
	m_panoramaUnits[m_idxCurrentGPU][panoUnitIdx]->unRegisterCUDAArray(weightmapCamIndex);

	m_context->functions()->glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 0, 0, w, h, 0);
	m_context->functions()->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_panoramaUnits[m_idxCurrentGPU][panoUnitIdx]->getDeltaWeightTexture(weightmapCamIndex), 0);
	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	m_panoramaUnits[m_idxCurrentGPU][panoUnitIdx]->registerCUDAArray(weightmapCamIndex);
}

void D360Stitcher::copyToUndoRedoBuffer(int panoUnitIdx, int weightmapCamIdx, WeightMapUndoRedo &undoRedo)
{
	GLuint texId;
	m_context->functions()->glGenTextures(1, &texId);
	m_context->functions()->glBindTexture(GL_TEXTURE_2D, texId);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	m_context->functions()->glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	int w = m_gaSettings->getCameraInput(weightmapCamIdx).xres;
	int h = m_gaSettings->getCameraInput(weightmapCamIdx).yres;
	m_context->functions()->glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, NULL);

	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, m_panoramaUnits[m_idxCurrentGPU][panoUnitIdx]->getDeltaWeightMapFrameBuffer(weightmapCamIdx));
	m_context->functions()->glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 0, 0, w, h, 0);
	m_context->functions()->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if (panoUnitIdx == 0)
		undoRedo.leftUndoRedoTexId = texId;
	else
		undoRedo.rightUndoRedoTexId = texId;
}

void D360Stitcher::clampUndoRedoBufferSize()
{
	while (m_undoRedo.size() > UNDOREDO_BUFFER_SIZE)
	{
		if (m_undoRedo[0].leftUndoRedoTexId != -1) {
			GLuint texId = m_undoRedo[0].leftUndoRedoTexId;
			glDeleteTextures(1, &texId);
		}
			
		if (m_undoRedo[0].rightUndoRedoTexId != -1) {
			GLuint texId = m_undoRedo[0].rightUndoRedoTexId;
			glDeleteTextures(1, &texId);
		}

		m_undoRedo.erase(m_undoRedo.begin());
	}
}

void D360Stitcher::deltaWeight2UndoRedo()
{
	WeightMapUndoRedo undoRedo;
	undoRedo.camIndex = m_weightMapCameraIndex;
	undoRedo.eyeMode = m_weightMapEyeMode;

	if (undoRedo.eyeMode == BOTHMODE) {
		for (int idx = 0; idx < 2; idx++) {
			if (checkAvailableIndex(idx, m_weightMapCameraIndex))
				copyToUndoRedoBuffer(idx, m_weightMapCameraIndex, undoRedo);
		}
	}
	else {
		int idx = undoRedo.eyeMode == RIGHTMODE ? 1 : 0;
		copyToUndoRedoBuffer(idx, m_weightMapCameraIndex, undoRedo);
	}

	m_undoRedo.push_back(undoRedo);
	m_undoRedoBufferIdx = m_undoRedo.size() - 1;

	clampUndoRedoBufferSize();
}

void D360Stitcher::undoRedo2DeltaWeight()
{
	if (m_undoRedo[m_undoRedoBufferIdx].eyeMode == BOTHMODE) {
		if (m_undoRedo[m_undoRedoBufferIdx].leftUndoRedoTexId != -1)
			copyFromUndoRedoBuffer(0, m_weightMapCameraIndex);
		
		if (m_undoRedo[m_undoRedoBufferIdx].rightUndoRedoTexId != -1)
			copyFromUndoRedoBuffer(1, m_weightMapCameraIndex);
	}
	else {
		int idx = (m_undoRedo[m_undoRedoBufferIdx].eyeMode == RIGHTMODE) ? 1 : 0;
		copyFromUndoRedoBuffer(idx, m_weightMapCameraIndex);
	}
}

void D360Stitcher::sendUndoRedoUIChangedToMainWindow()
{
	if (m_undoRedo.size() > 1 ) {
		if (m_undoRedoBufferIdx < m_undoRedo.size() - 1)
		{
			g_mainWindow->sendWeightMapEditRedoStatusChanged(true);
			if (m_undoRedoBufferIdx > 0)
				g_mainWindow->sendWeightMapEditUndoStatusChanged(true);
			else
				g_mainWindow->sendWeightMapEditUndoStatusChanged(false);
		}
		else
		{
			g_mainWindow->sendWeightMapEditRedoStatusChanged(false);
			g_mainWindow->sendWeightMapEditUndoStatusChanged(true);
		}
	}
	else {
		g_mainWindow->sendWeightMapEditUndoStatusChanged(false);
		g_mainWindow->sendWeightMapEditRedoStatusChanged(false);
	}
}

void D360Stitcher::makeCameraWeightMap()
{
	if (m_weightMapEyeMode == WeightMapEyeMode::BOTHMODE) {
		bool bAvailable = false;
		for (int i = 0; i < 2; i++) {
			if (bAvailable)
				break;
			bAvailable = checkAvailableIndex(i, m_weightMapCameraIndex);
		}
		if (!bAvailable)
		{
			m_weightMapChanged = false;
			return;
		}
	}
	else if (m_weightMapEyeMode == WeightMapEyeMode::MIRROR) {
		bool bAvailable = false;
		for (int i = 0; i < 2; i++){
			if (bAvailable)
				break;
			bAvailable = checkAvailableIndex(i, m_weightMapCameraIndex);
		}
		if (!bAvailable)
		{
			m_weightMapChanged = false;
			return;
		}
	}
	else{
		int idx = m_weightMapEyeMode == WeightMapEyeMode::RIGHTMODE ? 1 : 0;
		if (!checkAvailableIndex(idx, m_weightMapCameraIndex))
		{
			m_weightMapChanged = false;
			return;
		}
	}
	if (!m_isUndoRedo && m_undoRedo.size() <= 0) {
		deltaWeight2UndoRedo();
		sendUndoRedoUIChangedToMainWindow();
	}
	
	if (m_isUndoRedo && m_undoRedo.size() > 0) {
		m_isUndoRedo = false;
		m_weightmapCamIdxChangedByUndoRedo = false;
		if (m_weightMapCameraIndex != m_undoRedo[m_undoRedoBufferIdx].camIndex){
			m_weightMapCameraIndex = m_undoRedo[m_undoRedoBufferIdx].camIndex;
			g_mainWindow->sendWeightMapEditCameraIndexChanged(m_weightMapCameraIndex);
			m_weightmapCamIdxChangedByUndoRedo = true;
		}
		undoRedo2DeltaWeight();
		sendUndoRedoUIChangedToMainWindow();
		return;
	}

	if (m_weightmapCamIdxChanged && !m_isWeightMapReset) {	 
		if (!m_weightmapCamIdxChangedByUndoRedo) {
			deltaWeight2UndoRedo();
			sendUndoRedoUIChangedToMainWindow();
		}
	 	else
	 		m_weightmapCamIdxChangedByUndoRedo = false;

	 	return;
	}
	 
	if (m_undoRedoState == STARTING_UNDOREDO) {	 
	 	m_weightmapCamIdxChangedByUndoRedo = false;
	 	if (m_undoRedoBufferIdx < (int)m_undoRedo.size() - 1) {
	 		for (int i = m_undoRedoBufferIdx + 1; i < m_undoRedo.size(); i++)
	 		{
				if (m_undoRedo[i].leftUndoRedoTexId != -1) {
					GLuint texId = m_undoRedo[i].leftUndoRedoTexId;
					m_context->functions()->glDeleteTextures(1, &texId);
				}
				if (m_undoRedo[i].rightUndoRedoTexId != -1) {
					GLuint texId = m_undoRedo[i].rightUndoRedoTexId;
					m_context->functions()->glDeleteTextures(1, &texId);
				}
					
	 			m_undoRedo.erase(m_undoRedo.begin() + i);
	 			i--;
	 		}
	 	}
	 
		if (m_undoRedo.size() <= 0) {
			deltaWeight2UndoRedo();
			sendUndoRedoUIChangedToMainWindow();
		}
			
		m_undoRedoState = STARTED_UNDOREDO;
	}
	
	if (m_weightMapChanged && m_weightMapEditPoints.size() > 0) {
		m_weightMapEditPoints_ = m_weightMapEditPoints;

		float yaw = m_gaSettings->getCameraInput(m_weightMapCameraIndex).m_cameraParams.m_yaw;
		float pitch = m_gaSettings->getCameraInput(m_weightMapCameraIndex).m_cameraParams.m_pitch;
		float roll = m_gaSettings->getCameraInput(m_weightMapCameraIndex).m_cameraParams.m_roll;

		mat3 localM = getCameraViewMatrix(yaw, pitch, roll);

		mat3 globalM = mat3_id;
		vec3 u(m_gaSettings->m_fRoll * sd_to_rad, m_gaSettings->m_fPitch * sd_to_rad, m_gaSettings->m_fYaw * sd_to_rad);
		globalM.set_rot_zxy(u);
		mat3 invGlobalM;
		invert(invGlobalM, globalM);

		mat3 finalM;
		mult(finalM, localM, invGlobalM);

		vec2 pt = m_weightMapEditPoints[0];
		m_weightMapEditPoints.clear();

		mat3 finalM_;
		vec2 pt_;
		if (m_weightMapEyeMode == WeightMapEyeMode::MIRROR)
		{
			float yaw_ = m_gaSettings->getCameraInput(m_weightMapCameraIndex_).m_cameraParams.m_yaw;
			float pitch_ = m_gaSettings->getCameraInput(m_weightMapCameraIndex_).m_cameraParams.m_pitch;
			float roll_ = m_gaSettings->getCameraInput(m_weightMapCameraIndex_).m_cameraParams.m_roll;

			mat3 localM_ = getCameraViewMatrix(yaw_, pitch_, roll_);

			mat3 globalM_ = mat3_id;
			vec3 u_(m_gaSettings->m_fRoll * sd_to_rad, m_gaSettings->m_fPitch * sd_to_rad, m_gaSettings->m_fYaw * sd_to_rad);
			globalM_.set_rot_zxy(u_);
			mat3 invGlobalM_;
			invert(invGlobalM_, globalM_);

			mult(finalM_, localM_, invGlobalM_);

			pt_ = m_weightMapEditPoints_[0];
			m_weightMapEditPoints_.clear();
		}

		if (m_weightMapEyeMode == WeightMapEyeMode::BOTHMODE)
		{
			for (int idx = 0; idx < 2; idx++){
				if (checkAvailableIndex(idx, m_weightMapCameraIndex)){
					m_panoramaUnits[m_idxCurrentGPU][idx]->renderDeltaWeight(m_weightMapCameraIndex, m_weightMapRadius, m_weightMapFallOff, m_weightMapStrength, pt.x, pt.y, m_isWeightMapIncrement, finalM);
				}
			}
		}
		else if (m_weightMapEyeMode == WeightMapEyeMode::MIRROR)
		{
			for (int idx = 0; idx < 2; idx++){
				if (checkAvailableIndex(idx, m_weightMapCameraIndex)){
					if (idx == 0) {
						for (int idxPano = 0; idxPano < 2; idxPano++)
						{
							if (idxPano == 0)
								m_panoramaUnits[m_idxCurrentGPU][idxPano]->renderDeltaWeight(m_weightMapCameraIndex, m_weightMapRadius, m_weightMapFallOff, m_weightMapStrength, pt.x, pt.y, m_isWeightMapIncrement, finalM);
							else
								m_panoramaUnits[m_idxCurrentGPU][idxPano]->renderDeltaWeight(m_weightMapCameraIndex_, m_weightMapRadius, m_weightMapFallOff, m_weightMapStrength, pt_.x, pt_.y, m_isWeightMapIncrement, finalM_);
				}
			}
		}
		if (checkAvailableIndex(idx, m_weightMapCameraIndex_))
				{
					for (int idxPano = 0; idxPano < 2; idxPano++)
					{
						if (idxPano == 0)
							m_panoramaUnits[m_idxCurrentGPU][idxPano]->renderDeltaWeight(0, m_weightMapRadius, m_weightMapFallOff, m_weightMapStrength, pt.x, pt.y, m_isWeightMapIncrement, finalM);
						else
							m_panoramaUnits[m_idxCurrentGPU][idxPano]->renderDeltaWeight(m_weightMapCameraIndex_, m_weightMapRadius, m_weightMapFallOff, m_weightMapStrength, pt_.x, pt_.y, m_isWeightMapIncrement, finalM_);
					}
				}
			}
		}
		else
		{			int idx = m_weightMapEyeMode == WeightMapEyeMode::RIGHTMODE ? 1 : 0;
			m_panoramaUnits[m_idxCurrentGPU][idx]->renderDeltaWeight(m_weightMapCameraIndex, m_weightMapRadius, m_weightMapFallOff, m_weightMapStrength, pt.x, pt.y, m_isWeightMapIncrement, finalM);
		}
	}

	if (m_undoRedoState == ENDING_UNDOREDO) {
		deltaWeight2UndoRedo();
		sendUndoRedoUIChangedToMainWindow();
		
		m_undoRedoState = ENDED_UNDOREDO;
	}
}

void D360Stitcher::run()
{
	// Start timer ( used to calculate processing rate )
	t.start();
	int delay = 1000 / m_gaSettings->m_fps;
	int continueDelay = 5;
	
	while (1)
	{
		if (QThread::currentThread()->isInterruptionRequested())
		{
			doStop = true;
		}

		//
		// Stop thread if doStop = TRUE 
		//
		doStopMutex.lock();
		if (doStop)
		{
			doStop = false;
			doStopMutex.unlock();
			PANO_LOG("received Stitcher stop command");
			break;
		}
		doStopMutex.unlock();

		//PANO_LOG("Stitcher -- 1");
		sharedImageBuffer->waitStitcher();
		// for extra manipulation, we just restitch
		if (!isFirstFrame && doPause)
		{
			//PANO_LOG("Stitcher -- Pause");
			if (doReStitch)
			{
				doReStitch = false;
				if (m_context != QOpenGLContext::currentContext())
					m_context->makeCurrent(m_surface);
				updateCameraParams();
				setLutData(sharedImageBuffer->getGlobalAnimSettings()->getLutData());
				doStitch();
#ifdef USE_CUDA
				//cudaStreamSynchronize(g_CurStream);
#endif
				m_context->doneCurrent();
			}
			else if (doCalcGain)
			{
				doPauseMutex.lock();
				if (m_context != QOpenGLContext::currentContext())
					m_context->makeCurrent(m_surface);
				qDebug() << "Reset Gain...";
				calcExposure();
				m_context->doneCurrent();
				doCalcGain = false;
				doPauseMutex.unlock();
			}
			else if (doSnapshot && statsData.nFramesProcessed > 0)
			{
				doSnapshot = false;
				doPauseMutex.lock();
				QString snapshotPath = sharedImageBuffer->getGlobalAnimSettings()->m_snapshotDir;
				if (snapshotPath[snapshotPath.length() - 1] == '/')
					snapshotPath = snapshotPath.left(snapshotPath.length() - 1);
				QString imgName = QString(snapshotPath + "/pano%1_%2.jpg").arg(statsData.nFramesProcessed).arg(CUR_TIME_H);
				//Added By I
				int w = sharedImageBuffer->getGlobalAnimSettings()->getPanoXres();
				int h = sharedImageBuffer->getGlobalAnimSettings()->getPanoYres();

 				if (m_context != QOpenGLContext::currentContext())
 					m_context->makeCurrent(m_surface);
				saveTexture(getPanoramaFBO(), w, h, imgName, m_context->functions(), false);
 				m_context->doneCurrent();

				doPauseMutex.unlock();
				emit snapshoted();

				continue;
			}
			else
			{
				//
				// Pause thread if doPause = TRUE 
				//

				// This "IF" code is needed for fixing the issue that some cameras do NOT show any video screen sometimes, 
				// when firstFrame is captured 
				// (APP should pause when first frame is captured)
				if (cameraFrameRaw.size() == sharedImageBuffer->getSyncedCameraCount())
				{
					Sleep(100);
					continue;
				}
			}
		}
		else
		{
			//PANO_LOG("Stitcher -- Continue");
			if (cameraFrameProcN <= statsData.nFramesProcessed)
			{
				QThread::msleep(continueDelay);
				continue;
			}
			if (m_cudaThreadCount <= 1)
			{
				if (m_context != QOpenGLContext::currentContext())
					m_context->makeCurrent(m_surface);
				DoCompositePanorama(m_gaSettings);
				m_context->doneCurrent();
			}
			else
			{
#ifdef USE_CUDA
				m_idxCurrentGPU = statsData.nFramesProcessed % m_cudaThreadCount;
				cudaSetDevice(m_idxCurrentGPU);

				while (m_Thread[m_idxCurrentGPU].m_bIsRunning)
				{
					qApp->processEvents();
				}
				m_Thread[m_idxCurrentGPU].start();
#endif
			}
		}
	}

	PANO_LOG("Stopping stitcher thread (escaped from run function)...");
}

void D360Stitcher::DoCompositePanorama(GlobalAnimSettings* gasettings)
{
	static int avgTime = 0;
	static QTime avgTimer;

	t.restart();
#ifdef USE_CUDA
	if (g_CurStream == NULL && g_NextStream == NULL){
		g_CurStream = m_stream[0];
		g_NextStream = m_stream[1];
	}
#endif
	doCaptureIncomingFrames();

	if (isFirstFrame){
		setLutData(m_gaSettings->m_lutData);
#ifdef USE_CUDA
		for (int idx = 0; idx < m_cudaThreadCount; idx++){
			cudaSetDevice(idx);
			for (int i = 0; i < m_panoramaUnits[idx].size(); i++){
				m_panoramaUnits[idx][i]->updateGlobalParams(m_gaSettings->m_fYaw, m_gaSettings->m_fPitch, m_gaSettings->m_fRoll);
			}
		}
		cudaSetDevice(m_idxCurrentGPU);
#endif
	}
		

	sharedImageBuffer->wakeForVideoProcessing(statsData.nFramesProcessed);
	if (doReStitch)
	{
		doReStitch = false;
		updateCameraParams();
	}
	doStitch(gasettings->m_stitch);
	doOutput();
#ifdef USE_CUDA
	g_CurStream = m_stream[1];
	g_NextStream = m_stream[0];
#endif
	// Save processing time
	avgTime = avgTimer.elapsed();
	avgTimer.restart();

	// Update statistics
	updateFPS(avgTime);
	statsData.nFramesProcessed++;

	if (isFirstFrame)
		isFirstFrame = false;

	if (gasettings->m_ui == true)
	{
		emit updateStatisticsInGUI(statsData);
	}
}

void D360Stitcher::snapshot()
{
	QMutexLocker locker(&doPauseMutex);
	if (!doPause)	return;	
	doSnapshot = true;
}


void D360Stitcher::updateFPS(int timeElapsed)
{
	statsData.elapesedTime += timeElapsed; 

	// Add instantaneous FPS value to queue
	if (timeElapsed > 0)
	{
		fps.enqueue((int)1000 / timeElapsed);
		// Increment sample number
		sampleNumber++;
	}

	// Maximum size of queue is DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH
	if (fps.size() > STITCH_FPS_STAT_QUEUE_LENGTH)
		fps.dequeue();

	// Update FPS value every DEFAULT_PROCESSING_FPS_STAT_QUEUE_LENGTH samples
	if ((fps.size() == STITCH_FPS_STAT_QUEUE_LENGTH) && (sampleNumber == STITCH_FPS_STAT_QUEUE_LENGTH))
	{
		// Empty queue and store sum
		while (!fps.empty())
			fpsSum += fps.dequeue();
		// Calculate average FPS
		statsData.averageFPS = 1.0f * fpsSum / STITCH_FPS_STAT_QUEUE_LENGTH;
		// Reset sum
		fpsSum = 0;
		// Reset sample number
		sampleNumber = 0;
	}
}

void D360Stitcher::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex);
	finishMutex.unlock();
}

bool D360Stitcher::isFinished()
{
	return m_finished;
}

void D360Stitcher::playAndPause(bool isPause)
{
	QMutexLocker locker(&doPauseMutex);
	doPause = isPause;
	if (!isPause)
		t.restart();
}

void D360Stitcher::calcGain()
{
	QMutexLocker locker(&doPauseMutex);
	doCalcGain = true;
}

void D360Stitcher::resetGain()
{
	GlobalAnimSettings::CameraSettingsList& camsettings = g_mainWindow->getGlobalAnimSetting().cameraSettingsList();
	for (int i = 0; i < m_nViewCount; i++)
	{
		camsettings[i].exposure = camsettings[i].m_cameraParams.m_expOffset;
	}
	restitch();
}

void D360Stitcher::restitch(bool cameraParamChanged)
{
	QMutexLocker locker(&doPauseMutex);
	//if (doPause && !doReStitch)
	doReStitch = true;
	m_isCameraParamChanged = cameraParamChanged;
#ifdef USE_CUDA
	for (int idx = 0; idx < m_cudaThreadCount; idx++){
		cudaSetDevice(idx);
		for (int i = 0; i < m_panoramaUnits[idx].size(); i++){
			m_panoramaUnits[idx][i]->updateGlobalParams(m_gaSettings->m_fYaw, m_gaSettings->m_fPitch, m_gaSettings->m_fRoll);
		}
	}
#endif
	sharedImageBuffer->wakeStitcher();
}

void D360Stitcher::updateCameraParams()
{
	QMutexLocker locker(&doPauseMutex);
	for (int i = 0; i < m_viewProcessors[m_idxCurrentGPU].size(); i++)
	{
		m_viewProcessors[m_idxCurrentGPU][i]->updateCameraParams();
	}
	for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU].size(); i++)
	{
		m_panoramaUnits[m_idxCurrentGPU][i]->setCameraInput();
		m_panoramaUnits[m_idxCurrentGPU][i]->updateCameraParams();
	}
}

bool D360Stitcher::checkAvailableIndex(int eyeIndex, int cameraIndex)
{
	if (eyeIndex < m_panoramaUnits[m_idxCurrentGPU].size())
	{
		for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU][eyeIndex]->getGlobalViewIndexList().size(); i++)
		{
			if (cameraIndex == m_panoramaUnits[m_idxCurrentGPU][eyeIndex]->getGlobalViewIndex(i))
			{
				return true;
			}
		}
	}
	return false;
}

int D360Stitcher::getAvailableIndex(int eyeIndex, int cameraIndex)
{
	int retIdx = -1;

	if (eyeIndex < m_panoramaUnits[m_idxCurrentGPU].size()) {
		for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU][eyeIndex]->getGlobalViewIndexList().size(); i++)
		{
			if (cameraIndex == m_panoramaUnits[m_idxCurrentGPU][eyeIndex]->getGlobalViewIndex(i))
			{
				retIdx = i;
				break;
			}
			else
			{
				retIdx = -1;
			}
		}
	}
	else {
		retIdx = -1;
	}

	if (false/*g_mainWindow->m_eyeMode == MIRROR*/)
	{
		for (eyeIndex = 0; eyeIndex <  m_panoramaUnits[m_idxCurrentGPU].size(); eyeIndex ++)
		{
			for (int i = 0; i < m_panoramaUnits[m_idxCurrentGPU][eyeIndex]->getGlobalViewIndexList().size(); i++){
				if (cameraIndex == m_panoramaUnits[m_idxCurrentGPU][eyeIndex]->getGlobalViewIndex(i)){
					retIdx = i;
				}
			}
		}
	}

	return retIdx;
}



void D360Stitcher::updateWeightMapParams(WeightMapEyeMode eyeMode, int cameraIndex, int cameraIndex_,  float strength, int radius, float falloff, bool isIncrement, bool bChangeCameraIdx, int x, int y, bool isRightPos)
{
	//QMutexLocker locker(&doWeightmapUpdateParamMutex);

	m_weightmapOriginalCameraIndex = cameraIndex;
	m_weightMapCameraIndex = cameraIndex;
	m_weightMapCameraIndex_ = cameraIndex_;
	m_weightmapCamIdxChanged = bChangeCameraIdx;
	m_weightMapEyeMode = eyeMode;
	m_weightMapStrength = strength;
	m_weightMapRadius = radius;
	m_weightMapFallOff = falloff;
	m_isWeightMapIncrement = isIncrement;
	m_weightMapEditPoints.push_back(vec2(x, y));
	m_weightMapRightPos = isRightPos;

	restitch();	
}

void D360Stitcher::setWeightMapEditMode(bool isEditMode)
{
	if (isEditMode)
	{
		g_mainWindow->enableSeam(sharedImageBuffer->getSelectedView1(), sharedImageBuffer->getSelectedView2());
		g_mainWindow->setBlendMode(g_mainWindow->getBlendMode());

		sendUndoRedoUIChangedToMainWindow();
	}
	else
	{
		g_mainWindow->enableSeam(sharedImageBuffer->getSelectedView1(), sharedImageBuffer->getSelectedView2());
		g_mainWindow->setBlendMode(g_mainWindow->getBlendMode());
	}
	restitch();
}

void D360Stitcher::setWeightMapChanged(bool weightMapChanged)
{
	

	if (m_weightMapChanged == false && weightMapChanged == true)
	{
		m_undoRedoState = STARTING_UNDOREDO;
		restitch();
	}
	else if (m_weightMapChanged == true && weightMapChanged == false)
	{
		m_undoRedoState = ENDING_UNDOREDO;
		restitch();
	}
	else if (m_weightMapChanged == true && weightMapChanged == true) {
		//m_undoRedoState = STARTED_UNDOREDO;
	}
	else if (m_weightMapChanged == false && weightMapChanged == false) {
		//m_undoRedoState = ENDED_UNDOREDO;
	}

	m_weightMapChanged = weightMapChanged;
	if (!m_weightMapChanged)
	{
		m_weightMapEditPoints.clear();
	}

}

void D360Stitcher::resetEditedWeightMap()
{
	for (int k = 0; k < m_panoramaUnits[m_idxCurrentGPU].size(); k++)
	{
		m_panoramaUnits[m_idxCurrentGPU][k]->resetEditedWeightMap();
	}

	m_weightMapEditPoints.clear();

	for (int i = 0; i < m_undoRedo.size(); i++){
		GLuint texId = m_undoRedo[i].leftUndoRedoTexId;
		m_context->functions()->glDeleteTextures(1, &texId);
		texId = m_undoRedo[i].rightUndoRedoTexId;
		m_context->functions()->glDeleteTextures(1, &texId);
	}
	m_undoRedo.clear();
	m_undoRedoBufferIdx = 0;
	m_weightmapCamIdxChanged = false;
	m_weightmapCamIdxChangedByUndoRedo = false;
	m_isUndoRedo = false;
	m_undoRedoState = NONE_UNDOREDO;

	sendUndoRedoUIChangedToMainWindow();
}