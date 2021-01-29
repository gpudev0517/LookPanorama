#include "GPUComposite.h"
#include <Dense>
#include "common.h"
#include "QmlMainWindow.h"

using namespace Eigen;
extern QmlMainWindow *g_mainWindow;

BannerInfo::BannerInfo()
{
#ifdef USE_CUDA
	if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		billColorCvt = new CUDAUniColorCvt();
	}
	else
#endif
	{
		billColorCvt = new GLSLUniColorCvt();
	}
};

void BannerInfo::dispose()
{
	delete billColorCvt;
	frame.dispose();
};

GPUComposite::GPUComposite(QObject *parent) : QObject(parent)
, m_initialized(false)
, m_feathering(NULL)
, m_multibandBlending(NULL)
, m_visualizer(NULL)
, m_bannerRenderer(NULL)
{
	m_blendingMode = GlobalAnimSettings::MultiBandBlending;
	m_multiBandLevel = 0;
}

GPUComposite::~GPUComposite()
{
	if (m_initialized)
	{
		if (m_feathering)
		{
			delete m_feathering;
			m_feathering = NULL;
		}

		if (m_multibandBlending)
		{
			delete m_multibandBlending;
			m_multibandBlending = NULL;
		}

		if (m_visualizer)
		{
			delete m_visualizer;
			m_visualizer = NULL;
		}

		if (m_bannerRenderer)
		{
			delete m_bannerRenderer;
			m_bannerRenderer = NULL;
		}
	}
}

void GPUComposite::setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0)
{
	m_gl = gl;
	m_functions_2_0 = functions_2_0;
}

void GPUComposite::initialize(int nViewCount, int panoWidth, int panoHeight, GlobalAnimSettings::BlendingMode blendingMode, int multiBandLevel)
{
	recordingSnapshotPath = "";

	m_viewCount = nViewCount;
	m_panoramaWidth = panoWidth;
	m_panoramaHeight = panoHeight;
	g_panoramaWidth = panoWidth;
	g_panoramaHeight = panoHeight;

	m_blendingMode = blendingMode;
	m_multiBandLevel = multiBandLevel;

#ifdef USE_CUDA
	if ( g_useCUDA/*((QmlApplicationSetting *)g_mainWindow->applicationSetting())->useCUDA()*/){
		isRunningByCUDA = true;
		m_feathering = new CUDAFeathering();

		m_visualizer = new CUDAWeightVisualizer();
	}
	else
#endif
	{
		isRunningByCUDA = false;
		m_feathering = new GLSLFeathering();

		m_multibandBlending = new GLSLMultibandBlending();
		m_multibandBlending->setGL(m_gl, m_functions_2_0);
		m_multibandBlending->initialize(m_panoramaWidth, m_panoramaHeight, m_viewCount);

		m_visualizer = new GLSLWeightVisualizer();
	}
	
	m_feathering->setGL(m_gl, m_functions_2_0);
	m_feathering->initialize(m_panoramaWidth, m_panoramaHeight, m_viewCount);

	m_visualizer->setGL(m_gl);
	m_visualizer->initialize(m_panoramaWidth, m_panoramaHeight, m_viewCount);

	m_bannerRenderer = new BannerRenderer();
	m_bannerRenderer->setGL(m_gl, m_functions_2_0);
	m_bannerRenderer->initialize(m_panoramaWidth, m_panoramaHeight);

	canUseBanner = false;

	m_initialized = true;
}

// render composited image from frame buffer
void GPUComposite::render(WeightMapPaintMode paintMode, GPUResourceHandle *textures, GPUResourceHandle *weightMaps, GPUResourceHandle boundaryTextures[], int compositeID, int currentCamIndexforWeight, int eyeMode)
{
	if (m_blendingMode == GlobalAnimSettings::Feathering)
	{
		m_feathering->render(textures, weightMaps);
	}	
	else if (m_blendingMode == GlobalAnimSettings::MultiBandBlending)
		m_multibandBlending->render(textures, m_multiBandLevel, boundaryTextures, weightMaps);
	else if (m_blendingMode == GlobalAnimSettings::WeightMapVisualization)
		m_visualizer->render(paintMode, textures, weightMaps, compositeID, currentCamIndexforWeight, eyeMode);
}

int GPUComposite::getBlendTexture()
{
	GPUResourceHandle textureID = -1;
	if (m_blendingMode == GlobalAnimSettings::Feathering)
		textureID = m_feathering->getTargetGPUResource();
	else if (m_blendingMode == GlobalAnimSettings::MultiBandBlending)
		textureID = m_multibandBlending->getTargetGPUResource();
	else if (m_blendingMode == GlobalAnimSettings::WeightMapVisualization)
		textureID = m_visualizer->getTargetGPUResource();
	return textureID;
}

GPUResourceHandle GPUComposite::getTargetGPUResource()
{
	return m_bannerRenderer->getBannerTexture();
}

float getMean(VectorXf gains)
{
	float means = 0.0f;
	for (int i = 0; i < gains.rows(); i++)
	{
		means += gains(i, 0);
	}
	if (gains.rows() != 0)
		means /= gains.rows();
	return means;
}

std::vector<float> GPUComposite::getExposureData(GPUResourceHandle fbos[], int viewCnt)
{
	INTERSECT_DATA ** intersect_datas = new INTERSECT_DATA*[viewCnt];
	for (int i = 0; i < viewCnt; i++)
	{
		intersect_datas[i] = new INTERSECT_DATA[viewCnt];
	}
	
	// since datas[i,j].meanSrc1Intensity == datas[j,i].meanSrc2Intensity, and intersectPixelCnt is symmetric, we can speed up this calculation
	for (int i = 0; i < viewCnt; i++) {
		std::vector<INTERSECT_DATA> datas;
		intersect_datas[i][i].intersectPixelCnt = 0;
		intersect_datas[i][i].meanSrc1Intensity = 0;
		intersect_datas[i][i].meanSrc2Intensity = 0;
		for (int j = i + 1; j < viewCnt; j++) {
			intersect_datas[i][j] = getInterSectData(fbos[i], fbos[j]);
			intersect_datas[j][i].intersectPixelCnt = intersect_datas[i][j].intersectPixelCnt;
			intersect_datas[j][i].meanSrc1Intensity = intersect_datas[i][j].meanSrc2Intensity;
			intersect_datas[j][i].meanSrc2Intensity = intersect_datas[i][j].meanSrc1Intensity;
		}
	}

	std::vector<std::vector<float>> gain_a;
	std::vector<float> gain_b;
	float sigma_n = 10.0;
	float sigma_g = 0.1;
	for (int k = 0; k < viewCnt; k++) {
		std::vector<float> a_datas;
		for (int j = 0; j < viewCnt; j++) {
			float a = 0.0f;
			if (k == j) {
				for (int i = 0; i < viewCnt; i++) {
					if (i != k) {
						a += 2.0f / (sigma_n*sigma_n)
							* intersect_datas[i][k].intersectPixelCnt
							* (intersect_datas[k][i].meanSrc1Intensity*intersect_datas[k][i].meanSrc1Intensity);
					}
					a += intersect_datas[k][i].intersectPixelCnt / (sigma_g * sigma_g);
				}
			}
			else {
				a = -2 * intersect_datas[j][k].intersectPixelCnt / (sigma_n*sigma_n)
					* (intersect_datas[j][k].meanSrc1Intensity * intersect_datas[k][j].meanSrc1Intensity);
			}
			a_datas.push_back(a);
		}

		float b = 0;
		for (int i = 0; i < viewCnt; i++)
		{
			b += intersect_datas[k][i].intersectPixelCnt / (sigma_g*sigma_g);
		}
		gain_b.push_back(b);
		gain_a.push_back(a_datas);
	}

	for (int i = 0; i < viewCnt; i++)
		delete[] intersect_datas[i];
	delete[] intersect_datas;

	std::vector<float> gains;
	if (viewCnt > 0) {
		MatrixXf A(viewCnt, viewCnt);
		VectorXf b = VectorXf(viewCnt);
		for (int i = 0; i < viewCnt; i++) {
			for (int j = 0; j < viewCnt; j++) {
				A(i, j) = gain_a[i][j];
			}	
			b(i, 0) = gain_b[i];
		}
		VectorXf vGains = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
		float meanGain = getMean(vGains);
		for (int i = 0; i < m_viewCount; i++)
		{
			gains.push_back(vGains(i, 0) / meanGain);
		}
	}
	return gains;
}

INTERSECT_DATA GPUComposite::getInterSectData(GPUResourceHandle src1ID, GPUResourceHandle src2ID)
{

	INTERSECT_DATA intersectData;

	int myDataLength = m_panoramaWidth * m_panoramaHeight * 4;
	uchar *buffer1 = (uchar *)malloc(myDataLength);
	uchar *buffer2 = (uchar *)malloc(myDataLength);
#ifdef USE_CUDA
	if (isRunningByCUDA == true){
		cudaMemcpyFromArray(buffer1, (cudaArray *)src1ID, 0, 0, m_panoramaWidth * m_panoramaHeight * 4, cudaMemcpyDeviceToHost);
		cudaMemcpyFromArray(buffer2, (cudaArray *)src2ID, 0, 0, m_panoramaWidth * m_panoramaHeight * 4, cudaMemcpyDeviceToHost);
	}
	else
#endif
	{
		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, src1ID);
		m_gl->glReadPixels(0, 0, m_panoramaWidth, m_panoramaHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer1);

		m_gl->glBindFramebuffer(GL_FRAMEBUFFER, src2ID);
		m_gl->glReadPixels(0, 0, m_panoramaWidth, m_panoramaHeight, GL_RGBA, GL_UNSIGNED_BYTE, buffer2);
	}
	
	

	int intersectCnt = 0;
	float meanSrc1 = 0.0, meanSrc2 = 0.0;
	for (int i = 0; i < m_panoramaWidth; i++) {
		for (int j = 0; j < m_panoramaHeight; j++) {
			int idx = j * m_panoramaWidth * 4 + i * 4;
			if (buffer1[idx + 3] != 0 && buffer2[idx + 3] != 0) {
				meanSrc1 += 0.299 * (float)buffer1[idx] + 0.587 * (float)buffer1[idx + 1] + 0.114 * (float)buffer1[idx + 2];
				meanSrc2 += 0.299 * (float)buffer2[idx] + 0.587 * (float)buffer2[idx + 1] + 0.114 * (float)buffer2[idx + 2];
				intersectCnt++;
			}
		}
	}

	if (intersectCnt > 0) {
		intersectData.intersectPixelCnt = intersectCnt;
		intersectData.meanSrc1Intensity = meanSrc1 / intersectCnt;
		intersectData.meanSrc2Intensity = meanSrc2 / intersectCnt;
	}
	else {
		intersectData.intersectPixelCnt = 0;
		intersectData.meanSrc1Intensity = 0;
		intersectData.meanSrc2Intensity = 0;
	}
	free(buffer1);
	free(buffer2);
	return intersectData;
}

void GPUComposite::mixBanner(std::vector<BannerInfo*> bannerInputs)
{
	float width = m_panoramaWidth;
	float height = m_panoramaHeight;

	/// billing
	m_bannerRenderer->render(getBlendTexture(), bannerInputs);

	canUseBanner = true;
}

void GPUComposite::recordMaskMap(QString maskPath)
{
	recordingMutex.lock();
	recordingSnapshotPath = maskPath;
	renderCount = 0;
	recordingMutex.unlock();
}

void GPUComposite::onBlendSettingUpdated(GlobalAnimSettings::BlendingMode mode, int level)
{
	setBlendingMode(mode, level);
}