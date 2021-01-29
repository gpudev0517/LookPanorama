#include "SinglePanoramaUnit.h"

#include "D360Stitcher.h"
#include "define.h"
#include "ConfigZip.h"
#include "QmlMainWindow.h"

extern QmlMainWindow *g_mainWindow;

SinglePanoramaUnit::SinglePanoramaUnit(SharedImageBuffer *pSharedImageBuffer)
: sharedImageBuffer(pSharedImageBuffer)
, m_Name("SinglePanoramaUnit")
{
	m_seam = new GPUSeam();
	m_composite = new GPUComposite();

#ifdef USE_CUDA
	if (g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		isUseCuda = true;
	}
	else
#endif
	{
		isUseCuda = false;
	}
	if (isUseCuda){
#ifdef USE_CUDA
		m_nodalBlending = new CUDANodalBlending();
		m_postProcessing = new CUDAPanoramaPostProcessing();

		cudaMalloc(&panoViewTexture, 8 * sizeof(GPUResourceHandle));
		cudaMalloc(&weightTextures, 8 * sizeof(GPUResourceHandle));
#endif
	}
	else
	{
		m_nodalBlending = new GLSLNodalBlending();
		m_postProcessing = new GLSLPanoramaPostProcessing();

		panoViewTexture = new GPUResourceHandle[8];
		weightTextures = new GPUResourceHandle[8];
	}

	connect(sharedImageBuffer->getGlobalAnimSettings(), SIGNAL(fireEventBlendSettingUpdated(GlobalAnimSettings::BlendingMode, int)),
		m_composite, SLOT(onBlendSettingUpdated(GlobalAnimSettings::BlendingMode, int))
		);
	connect(sharedImageBuffer, SIGNAL(fireEventViewSelected(int, int)),
		this, SLOT(selectView(int, int)) );
}

SinglePanoramaUnit::~SinglePanoramaUnit()
{
	disconnect();

	if (isUseCuda){
#ifdef USE_CUDA
		cudaFree(panoViewTexture);
		cudaFree(weightTextures);
#endif
	}
	else
	{
		delete[] panoViewTexture;
		delete[] weightTextures;
	}

	delete m_seam;
	delete m_composite;
	delete m_postProcessing;
	delete m_nodalBlending;

	for (int i = 0; i < m_nViewCount; i++)
	{
		delete m_weights[i];
	}
	m_weights.clear();

	m_editedWeightDeltaTextureId.clear();
}

void SinglePanoramaUnit::setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3)
{
	m_gaSettings = sharedImageBuffer->getGlobalAnimSettings();
	m_nViewCount = m_gaSettings->getCameraCount();
	this->gl = gl;
	this->functions_2_0 = functions_2_0;

	m_seam->setGL(gl, functions_2_0);
	m_composite->setGL(gl, functions_2_0);
	m_postProcessing->setGL(gl, functions_2_0);
	m_nodalBlending->setGL(gl, functions_2_0);

	//For WeightMap Editing
	for (int i = 0; i < m_nViewCount; i++)
	{
		GPUPanoramaWeight* pPanoWeight = new GPUPanoramaWeight();
		pPanoWeight->setCameraInput(m_gaSettings->getCameraInput(i));
		pPanoWeight->setGL(gl, 0, functions_4_3);
		m_weights.push_back(pPanoWeight);
	}
}

void SinglePanoramaUnit::initialize(int panoWidth, int panoHeight, int curViewCount, int panoIndex,
	GPUResourceHandle panoColorTextures[], int nodalCameraCount, bool haveNodalMaskImage)
{
	m_seamChanged = false;
	this->curViewCount = curViewCount;
	this->panoIndex = panoIndex;

	if (panoIndex == 0)
	{
		for (int i = 0; i < curViewCount; i++)
		{
			int globalIndex = m_gaSettings->getLeftIndices()[i];
			index2GlobalIndex.push_back(globalIndex);
			m_leftStereoIndexMap[globalIndex + 1] = "Left";
		}
	}
	else
	{
		for (int i = 0; i < curViewCount; i++)
		{
			int globalIndex = m_gaSettings->getRightIndices()[i];			
			index2GlobalIndex.push_back(globalIndex);
			m_rightStereoIndexMap[globalIndex + 1] = "Right";
		}
	}

	m_seam->initialize(panoWidth, panoHeight, curViewCount);
	m_composite->initialize(curViewCount, panoWidth, panoHeight,
		m_gaSettings->m_blendingMode, m_gaSettings->m_multiBandLevel);
	m_postProcessing->initialize(panoWidth, panoHeight);

	// Nodal Shooting
	if (m_gaSettings->isNodalAvailable()) {
		m_nodalBlending->initialize(panoWidth, panoHeight, nodalCameraCount, haveNodalMaskImage);
	}

	for (int i = 0; i < m_nViewCount; i++)
	{
		int camWidth = m_gaSettings->getCameraInput(i).xres;
		int camHeight = m_gaSettings->getCameraInput(i).yres;

		m_weights[i]->initialize(camWidth, camHeight, panoWidth, panoHeight);
		m_editedWeightDeltaTextureId.push_back(m_weights[i]->getDeltaWeightTextureForUndoRedo());
		
		//load weightMap 
		int weightMapIndex = panoIndex * m_nViewCount + i;
		if (m_gaSettings->m_weightMaps.size() > weightMapIndex)
		{
			if (m_gaSettings->isINIfile)
			{
				WeightMapInput info = m_gaSettings->m_weightMaps[weightMapIndex];
				m_weights[i]->loadWeightmap(info.weightMapFile);
			}
			else
			{
				QString weightMapPath = QString(m_gaSettings->m_weightMapDir + "/weightMap%1_%2.png").arg(panoIndex).arg(i);
				if (QFile::exists(weightMapPath))
				{
					m_weights[i]->loadWeightmap(weightMapPath);
					QFile::remove(weightMapPath);
				}
			}
		}
	}

	GPUResourceHandle weightFboTextures[8];
	for (int i = 0; i < m_nViewCount; i++)
		weightFboTextures[i] = getPanoramaWeightTexture(i);

	GPUResourceHandle panoTex[8];
	GPUResourceHandle weightTex[8];
	for (int i = 0; i < curViewCount; i++)
	{
		int viewIdx = index2GlobalIndex[i];

		panoTex[i] = panoColorTextures[viewIdx];
		weightTex[i] = weightFboTextures[viewIdx];
	}
	
#ifdef USE_CUDA
	if (g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		cudaMemcpyAsync(panoViewTexture, panoTex, curViewCount * sizeof(GPUResourceHandle), cudaMemcpyHostToDevice);
		cudaMemcpyAsync(weightTextures, weightTex, curViewCount * sizeof(GPUResourceHandle), cudaMemcpyHostToDevice);
	}
	else
#endif
	for (int i = 0; i < curViewCount; i++)
	{
		panoViewTexture[i] = panoTex[i];
		weightTextures[i] = weightTex[i];
	}
}

void SinglePanoramaUnit::render(QList<int> nodalColorTextures, QList<int> nodalWeightTextures, std::vector<BannerInfo>& bannerInfos,
	WeightMapPaintMode paintMode, int weightCameraIndex, int eyeMode, vec3 ctLightColor)
{
	// composite
	GPUResourceHandle boundaryTextures[8];
	for (int j = 0; j < m_seam->getViewCount(); j++)
		boundaryTextures[j] = m_seam->getBoundaryTexture();
	m_composite->render(paintMode,
		panoViewTexture,
		weightTextures,
		boundaryTextures,
		panoIndex,
		weightCameraIndex, eyeMode //stitch
		);
	m_composite->canUseBanner = false;

	// mixBanner
	std::vector<BannerInfo*> curBannerInfo;
	for (int i = 0; i < bannerInfos.size(); i++)
	{
		BannerInfo& banner = bannerInfos[i];
		if (banner.isValid)
		{
			if (panoIndex == 0 && !banner.isStereoRight)
			{
				curBannerInfo.push_back(&banner);
			}
			if (panoIndex == 1 && banner.isStereoRight)
			{
				curBannerInfo.push_back(&banner);
			}
		}
	}
	m_composite->mixBanner(curBannerInfo);

	if (m_gaSettings->isNodalAvailable())
		m_nodalBlending->render(m_composite->getTargetGPUResource(), nodalColorTextures, nodalWeightTextures);
	
	int prePano = m_gaSettings->isNodalAvailable() ? m_nodalBlending->getTargetGPUResource() : m_composite->getTargetGPUResource();
	m_postProcessing->render(prePano, ctLightColor,
		m_gaSettings->m_fYaw, m_gaSettings->m_fPitch, m_gaSettings->m_fRoll,
		sharedImageBuffer->getSelectedView1() >= 0 ? m_seam->getSeamTexture() : -1);
}

void SinglePanoramaUnit::updateWeightMap(bool weightMapChanged, int globalViewIndex, std::vector<GPUResourceHandle> srcWeights)
{
	if (globalViewIndex >= 0)
		m_weights[globalViewIndex]->renderWeight(srcWeights[globalViewIndex], globalViewIndex);
	else if (globalViewIndex == -1)
	{
		for (int i = 0; i < m_weights.size(); i++)
		{
			m_weights[i]->renderWeight(srcWeights[i], i);
		}
	}
	m_seam->render(weightTextures, weightMapChanged);
}

void SinglePanoramaUnit::calcExposure(GPUResourceHandle fbos[])
{
	GlobalAnimSettings::CameraSettingsList& camsettings = m_gaSettings->cameraSettingsList();

	GPUResourceHandle curFbos[16];
	for (int i = 0; i < curViewCount; i++)
	{
		int globalIndex = index2GlobalIndex[i];
		curFbos[i] = fbos[globalIndex];
	}

	std::vector<float> gains = m_composite->getExposureData(curFbos, curViewCount);
	for (int i = 0; i < curViewCount; i++)
	{
		int globalIndex = index2GlobalIndex[i];
		camsettings[globalIndex].exposure = gain2ev(
			ev2gain(camsettings[globalIndex].exposure) * gains[i]);
	}
}

int SinglePanoramaUnit::getPanoramaTexture()
{
	return m_postProcessing->getTargetGPUResource();
}

int SinglePanoramaUnit::getPanoramaWeightTexture(int globalViewIndex)
{
	if (globalViewIndex < 0 || globalViewIndex >= m_weights.size())
		return -1;
	return m_weights[globalViewIndex]->getPanoWeightTexture();
}

int SinglePanoramaUnit::getDeltaWeightMapFrameBuffer(int globalViewIndex)
{
	return m_weights[globalViewIndex]->getDeltaWeightFrameBuffer();
}

int SinglePanoramaUnit::getDeltaWeightTexture(int globalViewIndex)
{
	return m_editedWeightDeltaTextureId[globalViewIndex];
}


void SinglePanoramaUnit::selectView(int viewIdx1, int viewIdx2)
{
	int idx = viewIdx1;
	int eyeMode = g_mainWindow->m_eyeMode;
	QString strStereoStatus;
	bool isLeft = false;
	bool isRight = false;

	if (m_leftStereoIndexMap.keys().size() > 0 )
	{
		if (m_leftStereoIndexMap.contains(viewIdx1)) {
			strStereoStatus = m_leftStereoIndexMap[viewIdx1];
			isLeft = true;
			idx = -1;
		}

		if (m_leftStereoIndexMap.contains(viewIdx2)) {
			strStereoStatus = m_leftStereoIndexMap[viewIdx2];
			isLeft = true;
			idx = -1;
		}
	}

	if (m_rightStereoIndexMap.keys().size() > 0)
	{
		if (m_rightStereoIndexMap.contains(viewIdx1)) {
			strStereoStatus = m_rightStereoIndexMap[viewIdx1];
			isRight = true;
			idx = -1;
		}

		if (m_rightStereoIndexMap.contains(viewIdx2)) {
			strStereoStatus = m_rightStereoIndexMap[viewIdx2];
			isRight = true;
			idx = -1;
		}
	}

	switch (eyeMode)
	{
	case WeightMapEyeMode::DEFAULT:
		break;
	case WeightMapEyeMode::LEFTMODE:
		if (strStereoStatus == "Left") {
			if (isLeft)
				for (int i = 0; i < m_leftStereoIndexMap.keys().size(); i++)
				{
					if (m_leftStereoIndexMap.keys()[i] == viewIdx1)
					{
						idx = i + 1;
						break;
					}
				}
			else if (isRight)
				idx = -1;
		}
		else
			idx = -1;			
		break;
	case WeightMapEyeMode::RIGHTMODE:
		if (strStereoStatus == "Right") {
			if (isRight)
				for (int i = 0; i < m_rightStereoIndexMap.keys().size(); i++)
				{
					if (m_rightStereoIndexMap.keys()[i] == viewIdx1)
					{
						idx = i + 1;
						break;
					}
				}
			else if (isLeft)
				idx = -1;
		}
		else
			idx = -1;
		break;
	case WeightMapEyeMode::BOTHMODE:
		if (strStereoStatus == "Left") {
			if (isLeft)
				for (int i = 0; i < m_leftStereoIndexMap.keys().size(); i++)
				{
					if (m_leftStereoIndexMap.keys()[i] == viewIdx1)
					{
						idx = i + 1;
						break;
					}
				}
		}
		else if (strStereoStatus == "Right") {
			if (isRight)
				for (int i = 0; i < m_rightStereoIndexMap.keys().size(); i++)
				{
					if (m_rightStereoIndexMap.keys()[i] == viewIdx1)
					{
						idx = i + 1;
						break;
					}
				}
		}
		else {
			idx = -1;
		}
		break;
	case MIRROR:
		if (strStereoStatus == "Left") {
			if (isLeft)
				for (int i = 0; i < m_leftStereoIndexMap.keys().size(); i++)
				{
					if (m_leftStereoIndexMap.keys()[i] == viewIdx1)
					{
						idx = i + 1;
						break;
					}
				}
		}
		else if (strStereoStatus == "Right") {
			if (isRight)
				for (int i = 0; i < m_rightStereoIndexMap.keys().size(); i++)
				{
					if (m_rightStereoIndexMap.keys()[i] == viewIdx2)
					{
						idx = i + 1;
						break;
					}
				}
		}
		break;
	default:
		break;
	}

// 	if (0 < viewIdx)
// 	{
// 		int view = viewIdx - 1;
// 		if (view < 0 || view >= index2GlobalIndex.size()) {
// 			idx = -1;
// 		} else {
// 			idx = index2GlobalIndex[view] + 1;
// 		}
// 	}

	m_seam->setSeamIndex(idx);
	sharedImageBuffer->getStitcher()->restitch();
}

void SinglePanoramaUnit::resetEditedWeightMap()
{
	for (int i = 0; i < m_editedWeightDeltaTextureId.size(); i++)
	{
		m_weights[i]->resetDeltaWeight();
	}
}

void SinglePanoramaUnit::saveWeightMaps()
{
	ConfigZip confZip;
	for (int i = 0; i < m_nViewCount; i++)
	{
		int weightMapIndex = panoIndex * m_nViewCount + i;
		QString fileName = m_gaSettings->m_weightMaps[weightMapIndex].weightMapFile;
		m_weights[i]->saveWeightmap(fileName);
		if (!confZip.createZipFile(m_gaSettings->m_weightMapDir, fileName))
		{
			PANO_N_LOG(QString("Weight map compression failed: %1").arg(fileName));
		}
	}
}

void SinglePanoramaUnit::renderDeltaWeight(int globalViewIndex, float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM)
{
	m_weights[globalViewIndex]->renderDelta(radius, falloff, strength, centerx, centery, increment, globalM, globalViewIndex);
}

void SinglePanoramaUnit::setCameraInput()
{
	for (int i = 0; i < m_nViewCount; i++)
	{
		m_weights[i]->setCameraInput(m_gaSettings->getCameraInput(i));
	}
}

void SinglePanoramaUnit::updateCameraParams()
{
	for (int i = 0; i < m_nViewCount; i++)
	{
		m_weights[i]->updateCameraParams();
	}
}

void SinglePanoramaUnit::updateGlobalParams(float yaw, float pitch, float roll)
{
	m_postProcessing->updateGlobalParams(yaw, pitch, roll);
}

int SinglePanoramaUnit::getGlobalViewIndex(int viewIndex)
{
	if (viewIndex < 0 || viewIndex >= index2GlobalIndex.size())
		return viewIndex;
	return index2GlobalIndex[viewIndex];
}