#ifndef GPUFINALPANORAMA_H
#define GPUFINALPANORAMA_H

#include "GPUProgram.h"
#include <QOpenGLBuffer>

/// <summary>
/// The final opengl shader that shows mono and stereo panorama in one view.
/// It can just show one panorama with the resolution of width by height,
/// or can show stereo panorama by top-bottom mode with the resolution of width by height*2.
/// </summary>
class GPUPanorama : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUPanorama(QObject *parent = 0);
	virtual ~GPUPanorama();

	virtual void initialize(int panoWidth, int panoHeight, bool isStereo) = 0;
	virtual void render(GPUResourceHandle fbos[], bool isOutput) = 0;

	virtual void downloadTexture(unsigned char* alphaBuffer) = 0;

	virtual GPUResourceHandle getTargetGPUResourceForInteract(){ return m_fboTextureId; }


	const int getWidth();
	const int getHeight();

protected:
	int panoramaViewWidth;
	int panoramaViewHeight;
	
	bool m_stereo;
};

struct FinalPanoramaConfig
{
	FinalPanoramaConfig(int width, int height, int stereo)
	{
		panoWidth = width;
		panoHeight = height;
		isStereo = stereo;
	}
	int panoWidth;
	int panoHeight;
	bool isStereo;
};

class GPUFinalPanorama : public GPUProgram
{
	Q_OBJECT
public:
	explicit GPUFinalPanorama(QObject *parent = 0);
	virtual ~GPUFinalPanorama();

	virtual void initialize(int panoWidth, int panoHeight, bool isStereo) = 0;
	virtual void requestReconfig(int panoWidth, int panoHeight, bool isStereo);
	virtual void render(GPUResourceHandle fbo) = 0;

	virtual void downloadTexture(unsigned char* rgbBuffer) = 0;

	const int getWidth();
	const int getHeight();

protected:
	int panoramaViewWidth;
	int panoramaViewHeight;

	int m_workingGPUResource;

	int m_targetArrayIndex;

	bool m_stereo;

	std::vector<FinalPanoramaConfig> newConfig;
};

#endif // GPUFINALPANORAMA_H