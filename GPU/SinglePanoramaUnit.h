#pragma once
#include <QObject>

#include "SharedImageBuffer.h"

#include "GPUSeam.h"
#include "GPUComposite.h"
#include "GPUPanoramaWeight.h"
#include "GLSLNodalBlending.h"

#ifdef USE_CUDA
#include "CUDAPanoramaPostProcessing.h"
#endif
#include "GLSLPanoramaPostProcessing.h"
#include <QOpenGLFunctions_4_3_Compatibility>

class SinglePanoramaUnit : public QObject
{
	Q_OBJECT
public:
	SinglePanoramaUnit(SharedImageBuffer *pSharedImageBuffer);
	virtual ~SinglePanoramaUnit();
	void setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3);
	void initialize(int panoWidth, int panoHeight, int curViewCount, int panoIndex, GPUResourceHandle panoColorTextures[], int nodalCameraCount, bool haveNodalMaskImage);
	void render(QList<int> nodalColorTextures, QList<int> nodalWeightTextures, std::vector<BannerInfo>& banners,
		WeightMapPaintMode paintMode, int weightCameraIndex, int eyeMode, vec3 ctLightColor);
	void updateWeightMap(bool weightMapChanged, int globalViewIndex, std::vector<GPUResourceHandle> srcWeights);
	void calcExposure(GPUResourceHandle fbos[]);

	int getPanoramaTexture();
	int getPanoramaWeightTexture(int globalViewIndex);

	int getDeltaWeightMapFrameBuffer(int globalViewIndex);
	int getDeltaWeightTexture(int globalViewIndex);
	void resetEditedWeightMap();
	void saveWeightMaps();

	void renderDeltaWeight(int globalViewIndex, float radius, float falloff, float strength, float centerx, float centery, bool increment, mat3 &globalM);

	void setCameraInput();
	void updateCameraParams();
	void updateGlobalParams(float yaw, float pitch, float roll);

	void setLutData(QVariantList *vList){

		m_postProcessing->setLutData(vList);
	}

	virtual void refreshCUDAArray(int idx){
		m_weights[idx]->refreshCUDAArray();
	}

	void unRegisterCUDAArray(int idx){
		m_weights[idx]->unRegisterCUDAArray();
	}

	void registerCUDAArray(int idx){
		m_weights[idx]->registerCUDAArray();
	}
	// Utility
	int getGlobalViewIndex(int viewIndex);
	std::vector<int> &getGlobalViewIndexList(){
		return index2GlobalIndex;
	}

private:
	QOpenGLFunctions* gl;
	QOpenGLFunctions_2_0* functions_2_0;
	bool isUseCuda;

	SharedImageBuffer* sharedImageBuffer;
	GlobalAnimSettings* m_gaSettings;

	GPUSeam* m_seam;
	GPUComposite* m_composite;
	GPUPanoramaPostProcessing* m_postProcessing;
	GPUNodalBlending* m_nodalBlending;

	// Weight map
	std::vector<GPUPanoramaWeight*> m_weights;
	std::vector<unsigned int> m_editedWeightDeltaTextureId;

	// Framebuffer texture for unwarped textures [8]
	GPUResourceHandle* panoViewTexture;
	GPUResourceHandle* weightTextures;

	// property
	int curViewCount; // view count for the current pano
	int m_nViewCount; // total view count
	int panoIndex;

	// internal status
	std::vector<int> index2GlobalIndex; // cur view index to global view index
	QMap<int, QString> m_leftStereoIndexMap;
	QMap<int, QString> m_rightStereoIndexMap;

	// status
	bool m_seamChanged;

	QString m_Name;

public slots:
	void selectView(int viewIdx1, int viewIdx2);
};