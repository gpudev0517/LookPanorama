#include "GPUColorCvt.h"

#include "GLSLColorCvt.h"
#ifdef USE_CUDA
#include "CUDAColorCvt.h"
#endif
#include "common.h"


GPUColorCvt_2RGBA::GPUColorCvt_2RGBA(QObject *parent) : GPUProgram(parent)
{
}

GPUColorCvt_2RGBA::~GPUColorCvt_2RGBA()
{
}

const int GPUColorCvt_2RGBA::getWidth()
{
	return imageWidth;
}

const int GPUColorCvt_2RGBA::getHeight()
{
	return imageHeight;
}


void GPUColorCvt_2RGBA::render(QImage& img)
{
	ImageBufferData frame;

	int width = img.width();
	int height = img.height();
	qDebug() << "Weight format " << img.format();
	if (img.format() == QImage::Format_RGB888)
	{
		frame.mFormat = ImageBufferData::RGB888;
		frame.mImageY.stride = width * 3;
	}
	else if (img.format() == QImage::Format_ARGB32)
	{
		frame.mFormat = ImageBufferData::RGBA8888;
		frame.mImageY.stride = width * 4;
	}
	else if (img.format() == QImage::Format_Grayscale8)
	{
		frame.mFormat = ImageBufferData::GRAYSCALE;
		frame.mImageY.stride = width;
	}
	frame.mImageY.buffer = img.bits();
	frame.mImageY.width = width;
	frame.mImageY.height = height;
	
	frame.mImageU = AlignedImage();
	frame.mImageV = AlignedImage();

	render(frame);
}

void GPUColorCvt_2RGBA::renderEmptyFrame()
{
	/*ImageBufferData img;
	img.mImageV.width = this->imageWidth;
	img.mImageV.height = this->imageHeight;
	img.mImageY.stride = this->imageWidth * 3;

	img.mImageY.buffer = new unsigned char[img.mImageV.height * img.mImageY.stride];
	render(img, false, 0.0f);
	delete[] img.mImageY.buffer;*/
}

int GPUColorCvt_2RGBA::getReadyGPUResourceIndex()
{
	int textureIndex = m_workingGPUResource - 1;
	if (textureIndex < 0)
		textureIndex = textureIndex + m_targetCount;
	return textureIndex;
}

///
GPUUniColorCvt::GPUUniColorCvt()
{
	mConverter = NULL;
	mFormat = -1;
}

GPUUniColorCvt::~GPUUniColorCvt()
{
	Free();
}

void GPUUniColorCvt::Free()
{
	if (mConverter != NULL)
	{
		delete mConverter;
		mConverter = NULL;
		mFormat = -1;
	}
}

GPUResourceHandle GPUUniColorCvt::getTargetGPUResource() const
{
	if (mConverter != NULL)
		return mConverter->getTargetGPUResource();
	return 0;
}

bool GPUUniColorCvt::isRenderable() const
{
	return (mConverter != NULL);
}

// Alpha Texture
GPUAlphaSource::GPUAlphaSource( QObject *parent) : GPUProgram(parent)
{
}

GPUAlphaSource::~GPUAlphaSource()
{
}

int GPUAlphaSource::getReadyGPUResourceIndex()
{
	int textureIndex = m_workingGPUResource - 1;
	if (textureIndex < 0)
		textureIndex = textureIndex + m_targetCount;
	return textureIndex;
}

// Nodal Input
GPUNodalInput::GPUNodalInput()
 : colorCvt(NULL)
, weightMap(NULL)
{
	
}

GPUNodalInput::~GPUNodalInput()
{
	destroyColorCvt();
}


bool GPUNodalInput::isColorCvtReady()
{
	return colorCvt != NULL;
}

void GPUNodalInput::destroyColorCvt()
{
	if (colorCvt)
	{
		delete colorCvt;
		colorCvt = NULL;
	}

	if (weightMap)
	{
		delete weightMap;
		weightMap = NULL;
	}
}

GPUResourceHandle GPUNodalInput::getColorCvtGPUResource()
{
	if (isColorCvtReady())
	{
		return colorCvt->getTargetGPUResource();
	}
	return -1;
}

GPUResourceHandle GPUNodalInput::getWeightGPUResource()
{
	if (weightMap)
		return weightMap->getTargetGPUResource();
	return -1;
}


GPUColorCvt_2RGBA* GPUColorCvt_2RGBA::createColorCvt(int pixel_format, bool useCuda)
{
	if (pixel_format == AV_PIX_FMT_YUV420P || pixel_format == AV_PIX_FMT_YUVJ420P)
	{
#ifdef USE_GPU_COLORCVT
#ifdef USE_CUDA
		if (useCuda){
			return new CUDAColorCvt_YUV2RGBA();
		}
		else
#endif
		{
			return new GLSLColorCvt_YUV2RGBA();
		}
	}
	else if (pixel_format == AV_PIX_FMT_YUYV422)
	{
#ifdef USE_CUDA
		if (useCuda){
			return new CUDAColorCvt_YUV4222RGBA();
		}
		else
#endif
		{
			return new GLSLColorCvt_YUV4222RGBA();
		}
	}
	else if (pixel_format == AV_PIX_FMT_YUVJ422P)
	{
#ifdef USE_CUDA
		if (useCuda){
			return new CUDAColorCvt_YUVJ422P2RGBA();
		}
		else
#endif
		{
			return new GLSLColorCvt_YUVJ422P2RGBA();
		}
	}
	else if (pixel_format == AV_PIX_FMT_BGR0)
	{
#ifdef USE_CUDA
		if (useCuda) {
			return new CUDAColorCvt_BGR02RGBA();
		}
		else
#endif
		{
			return new GLSLColorCvt_BGR02RGBA();
			//return new GLSLColorCvt_RGB2RGBA();
		}
	}
	else if (pixel_format == AV_PIX_FMT_BAYER_RGGB8)
	{
//#ifdef USE_CUDA
//		if (useCuda) {
//			return new CUDAColorCvt_BGR02RGBA();
//		}
//		else
//#endif
		{
			return new GLSLColorCvt_BAYERRG82RGBA();
			//return new GLSLColorCvt_RGB2RGBA();
		}
	}
#endif
	else
	{
#ifdef USE_CUDA
		if (useCuda){
			return new CUDAColorCvt_RGB2RGBA();
		}
		else
#endif
		{
			return new GLSLColorCvt_BAYERRG82RGBA();
			//return new GLSLColorCvt_RGB2RGBA();
		}
	}
}