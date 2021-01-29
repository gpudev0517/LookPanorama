#ifndef GPUCOMPOSITE_H
#define GPUCOMPOSITE_H

#include <QObject>

#include <QtGui/QOpenGLShaderProgram>
#include <QtGui/QOpenGLFunctions>
#include <QtGui/QOpenGLFunctions_2_0>
#include <QOpenGLTexture>

#include "common.h"
#include "D360Parser.h"

#include "GLSLFeathering.h"
#include "GLSLMultibandBlending.h"
#include "GLSLWeightVisualizer.h"
#include "GLSLBanner.h"

#ifdef USE_CUDA
#include "CUDAFeathering.h"
#include "CUDAWeightVisualizer.h"
#include "CUDANodalBlending.h"
#include "CUDABanner.h"
#endif

typedef struct {
	int intersectPixelCnt;
	int meanSrc1Intensity;
	int meanSrc2Intensity;
 } INTERSECT_DATA;

class GPUComposite : public QObject
{
    Q_OBJECT
public:
	explicit GPUComposite(QObject *parent = 0);
	virtual ~GPUComposite();

    void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0);
	void initialize(int nViewCount, int panoWidth, int panoHeight, GlobalAnimSettings::BlendingMode blendingMode, int multiBandLevel);
	void render(WeightMapPaintMode paintMode, GPUResourceHandle *textures, GPUResourceHandle *weightMaps, GPUResourceHandle boundaries[], int compositeID, int currentCameraIndex, int eyeMode);

	int getBlendTexture();
	GPUResourceHandle getTargetGPUResource();
	int getPanoramaWidth() { return m_panoramaWidth; }
	int getPanoramaHeight() { return m_panoramaHeight; }

	std::vector<float> getExposureData(GPUResourceHandle fbos[], int viewCnt);

	void setBlendingMode(GlobalAnimSettings::BlendingMode mode, int multiBandLevel = 0)
	{
		m_blendingMode = mode;
		if (mode == GlobalAnimSettings::MultiBandBlending)
			m_multiBandLevel = multiBandLevel;
	}

	void setBlendingModeWithoutLevel(GlobalAnimSettings::BlendingMode mode)
	{
		m_blendingMode = mode;
	}

	GlobalAnimSettings::BlendingMode getBlendingMode(){ return m_blendingMode; }

	void recordMaskMap(QString maskPath);


#define BANNER_FIRST 0x01
#define BANNER_LAST 0x02
	void mixBanner(std::vector<BannerInfo*> bannerInputs);
private:
	void multibandBlending(GPUResourceHandle fboTextures[], int bandLevel);
	INTERSECT_DATA getInterSectData(GPUResourceHandle src1, GPUResourceHandle src2);

private:
	int m_panoramaWidth, m_panoramaHeight, m_viewCount;	

	QOpenGLFunctions* m_gl;
	QOpenGLFunctions_2_0* m_functions_2_0;

	bool m_initialized;

	GPUFeathering* m_feathering;
	GLSLMultibandBlending* m_multibandBlending;
	GPUWeightVisualizer* m_visualizer;
	BannerRenderer* m_bannerRenderer;

	GlobalAnimSettings::BlendingMode m_blendingMode;
	int m_multiBandLevel;

public:
	bool canUseBanner = false; // flag to check banner-rendered successfully.
private:
	QMutex recordingMutex;
	int renderCount;
	QString recordingSnapshotPath;

	bool isRunningByCUDA;

public slots:
	void onBlendSettingUpdated(GlobalAnimSettings::BlendingMode mode, int level);
};

#endif // GPUCOMPOSITE_H