#ifndef CUDAPANO_POSTPROCESSING_H
#define CUDAPANO_POSTPROCESSING_H

#ifdef USE_CUDA
#include "GPUProgram.h"
#include "3DMath.h"
#include "GPUPanoramaPostProcessing.h"

class CUDAPanoramaPostProcessing : public GPUPanoramaPostProcessing
{
	Q_OBJECT
public:
	explicit CUDAPanoramaPostProcessing(QObject *parent = 0);
	virtual ~CUDAPanoramaPostProcessing();

	virtual void initialize(int panoWidth, int panoHeight);
	void render(GLuint panoTextureId, vec3 ctLightColor,
		float yaw, float pitch, float roll,
		int seamTextureId);

	virtual void setLutData(QVariantList *vList);
	virtual void updateGlobalParams(float yaw, float pitch, float roll);

private:

	cudaArray *m_lutArray;
	cudaSurfaceObject_t m_lutSurface;

	float *m_devPlaceMat;

};

#endif //USE_CUDA

#endif // CUDAPANO_COLORCORRECTION_H