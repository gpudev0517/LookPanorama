#ifndef GPUCOLORCVT_H
#define GPUCOLORCVT_H

#include <QtGui>

#include "GPUProgram.h"
#include "D360Parser.h"
#include "SharedImageBuffer.h"
#include "define.h"

class GPUColorCvt_2RGBA : public GPUProgram
{
	Q_OBJECT
public:
	GPUColorCvt_2RGBA(QObject *parent = 0);
	virtual ~GPUColorCvt_2RGBA();
	virtual void initialize(int imgWidth, int imgHeight) = 0;
	virtual void render(ImageBufferData& img) = 0;
	virtual void render(QImage& img);

	static GPUColorCvt_2RGBA* createColorCvt(int src_pixel_format, bool useCuda);

protected:

	virtual void createSourceGPUResource() = 0;
	virtual void setSourceGPUResourceBuffer(int index, const uchar* buf, bool textureChanged) = 0;

	virtual bool isColorConversionNeeded() = 0;
	void renderEmptyFrame();

	int getReadyGPUResourceIndex();

public:
	int ImageWidth() const { return imageWidth; }
	int ImageHeight() const { return imageHeight; }

	const int getWidth();
	const int getHeight();

protected:
	int imageWidth;
	int imageHeight;
	int imageBytesPerLine;

	static const int m_targetCount = 2;
	int m_targetArrayIndex;
	int m_workingGPUResource;
};

class GPUUniColorCvt
{
protected:
	GPUColorCvt_2RGBA* mConverter;
	int mFormat;
public:
	GPUUniColorCvt();
	virtual ~GPUUniColorCvt();
	void Free();
	// render
	virtual bool DynRender(ImageBufferData& img, void *gl = NULL) = 0;
	virtual GPUResourceHandle getTargetGPUResource() const;
	bool isRenderable() const;
	int ImageWidth() const
	{
		if (mConverter)
			return mConverter->ImageWidth();
		else
			return 0;
	}
	int ImageHeight() const
	{
		if (mConverter)
			return mConverter->ImageHeight();
		else
			return 0;
	}
};

// Alpha Texture
class GPUAlphaSource : public GPUProgram
{
public:

	GPUAlphaSource(QObject *parent = NULL);
	virtual ~GPUAlphaSource();

	virtual void initialize(int imgWidth, int imgHeight) = 0;
	virtual void render(QImage& img) = 0;

protected:
	virtual void createSourceGPUResource() = 0;
	virtual void setSourceGPUResourceBuffer(const uchar* bits, bool textureChanged) = 0;

	int getReadyGPUResourceIndex();

public:
	int ImageWidth() const { return imageWidth; }
	int ImageHeight() const { return imageHeight; }
protected:
	int imageWidth;
	int imageHeight;
	int imageBytesPerLine;

	static const int m_targetCount = 2;
	int m_targetArrayIndex;
	int m_workingGPUResource;

	const uchar* buffer_ptr;
};

// Nodal Input

class GPUNodalInput
{
protected:
	GPUColorCvt_2RGBA * colorCvt;
	GPUAlphaSource * weightMap;
	QImage weightImage;
	int weightRenderCount;
public:

	GPUNodalInput();
	virtual ~GPUNodalInput();
	
	virtual void createColorCvt(void* gl, bool liveMode, QString weightFilename) = 0;
	virtual void initialize(int imgWidth, int imgHeight) = 0;
	bool isColorCvtReady();
	virtual void render(ImageBufferData& img) = 0;
	GPUResourceHandle getColorCvtGPUResource();
	GPUResourceHandle getWeightGPUResource();
	void destroyColorCvt();


};

#endif // GPUCOLORCVT_H