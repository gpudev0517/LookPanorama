#ifndef GPUPANO_POSTPROCESSING_H
#define GPUPANO_POSTPROCESSING_H

#include "GPUProgram.h"

class GPUPanoramaPostProcessing : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUPanoramaPostProcessing(QObject *parent = 0);
	virtual ~GPUPanoramaPostProcessing();

	virtual void initialize(int panoWidth, int panoHeight) = 0;
	virtual void render(GLuint panoTextureId, vec3 ctLightColor,
		float yaw, float pitch, float roll,
		int seamTextureId) = 0;

	virtual const int getWidth();
	virtual const int getHeight();
	virtual void setLutData(QVariantList *vList) = 0;
	virtual void updateGlobalParams(float yaw, float pitch, float roll) = 0;
protected:
	int panoramaWidth;
	int panoramaHeight;
};

#endif // GPUPANO_POSTPROCESSING_H