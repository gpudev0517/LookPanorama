
#ifdef USE_CUDA

#include "CUDAColorCvt.h"
#include "common.h"

extern "C" void runColorCvt_YUV2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
extern "C" void runColorCvt_YUV4222RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
extern "C" void runColorCvt_YUVJ422P2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
extern "C" void runColorCvt_BGR02RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
extern "C" void runColorCvt_RGB2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);
extern "C" void runColorCvt_RGBA2RGBA_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline);

extern cudaStream_t g_CurStream;
extern cudaStream_t g_NextStream;

CUDAColorCvt_2RGBA::CUDAColorCvt_2RGBA(QObject *parent) : GPUColorCvt_2RGBA(parent)
{
	m_cudaSrcArrays[0] = m_cudaSrcArrays[1] = m_cudaTargetArrays[0] = m_cudaTargetArrays[1] = NULL;
	m_cudaSrcSurfaces[0] = m_cudaSrcSurfaces[1] = m_cudaSrcTextures[0] = m_cudaSrcTextures[1] = m_cudaTargetSurfaces[0] = m_cudaTargetSurfaces[1] = m_cudaTargetTextures[0] = m_cudaTargetTextures[0] = - 1;
}

CUDAColorCvt_2RGBA::~CUDAColorCvt_2RGBA()
{
	if (m_initialized)
	{
		for (int i = 0; i < m_targetCount; i++){
			if (m_cudaTargetArrays[i]){
				cudaFreeArray(m_cudaTargetArrays[i]);
				cudaDestroySurfaceObject(m_cudaTargetSurfaces[i]);
				cudaDestroyTextureObject(m_cudaTargetTextures[i]);
			}

		}

		for (int i = 0; i < m_targetCount; i++){
			if (m_cudaSrcArrays[i]){
				cudaFreeArray(m_cudaSrcArrays[i]);
				cudaDestroySurfaceObject(m_cudaSrcSurfaces[i]);
				cudaDestroyTextureObject(m_cudaSrcTextures[i]);
			}
		}
	}
}


void CUDAColorCvt_2RGBA::initialize(int imgWidth, int imgHeight)
{
	imageWidth = 0;
	imageHeight = 0;
	imageBytesPerLine = 0;

	for (int i = 0; i < m_targetCount; i++)
	{
 		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
 		cudaMallocArray(&m_cudaTargetArrays[i], &channelFormat, imgWidth, imgHeight, cudaArraySurfaceLoadStore);

		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = m_cudaTargetArrays[i];
		cudaCreateSurfaceObject(&m_cudaTargetSurfaces[i], &surfRes);

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;

		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.readMode = cudaReadModeNormalizedFloat;

		cudaCreateTextureObject(&m_cudaTargetTextures[i], &surfRes, &texDescr, NULL);
	}
	m_workingGPUResource = 0;
	m_targetArrayIndex = 0;

	m_initialized = true;

	renderEmptyFrame();
}


void CUDAColorCvt_2RGBA::render(ImageBufferData& img)
{
	//const uchar * bits = img.constBits();
	int width = img.mImageY.width;
	int height = img.mImageY.height;

	if (width == 0 || height == 0) return;

	bool blTextureFormatChanged = false;
	if (imageWidth != width || imageHeight != height)
	{
		imageWidth = width;
		imageHeight = height;
		imageBytesPerLine = img.mImageY.stride;

		createSourceGPUResource();
		blTextureFormatChanged = true;
	}


	const unsigned char * buffers[3] = { img.mImageY.buffer, img.mImageU.buffer, img.mImageV.buffer };
	m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
	for (int i = 0; i < 3; i++)
	{
		// host to device buffer uploading.
		setSourceGPUResourceBuffer(i, buffers[i], blTextureFormatChanged);


	}
	//qDebug("GLSLColorCvt_2RGB - 4");
	// Don't comment below code block absolutely.
	if (blTextureFormatChanged)
	{	// It's very important to avoid empty-rendering.
		m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
		for (int i = 0; i < 3; i++)
		{	// host to device buffer uploading.
			setSourceGPUResourceBuffer(i, buffers[i], false);

		}
	}
	
	if (blTextureFormatChanged)
	{	// It's very important to avoid empty-rendering.

		for (int i = 0; i < m_targetCount; i++){
			cudaResourceDesc    surfRes;
			memset(&surfRes, 0, sizeof(cudaResourceDesc));
			surfRes.resType = cudaResourceTypeArray;
			surfRes.res.array.array = m_cudaSrcArrays[i];
			cudaCreateSurfaceObject(&m_cudaSrcSurfaces[i], &surfRes);

			cudaTextureDesc             texDescr;
			memset(&texDescr, 0, sizeof(cudaTextureDesc));

			texDescr.normalizedCoords = 1;
			texDescr.filterMode = cudaFilterModeLinear;

			texDescr.addressMode[0] = cudaAddressModeClamp;
			texDescr.addressMode[1] = cudaAddressModeClamp;
			texDescr.addressMode[2] = cudaAddressModeClamp;

			texDescr.readMode = cudaReadModeNormalizedFloat;

			cudaCreateTextureObject(&m_cudaSrcTextures[i], &surfRes, &texDescr, NULL);
		}
	}
	runKernel(m_cudaTargetSurfaces[m_workingGPUResource], m_cudaSrcTextures[m_workingGPUResource], width, height, imageBytesPerLine);
	m_workingGPUResource = (m_workingGPUResource + 1) % m_targetCount;
}

//////////////

CUDAColorCvt_YUV2RGBA::CUDAColorCvt_YUV2RGBA(QObject *parent) : CUDAColorCvt_2RGBA(parent)
{
}

CUDAColorCvt_YUV2RGBA::~CUDAColorCvt_YUV2RGBA()
{

}

void CUDAColorCvt_YUV2RGBA::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageBytesPerLine, imageHeight * 3 / 2, cudaArraySurfaceLoadStore);
	}
	
}

void CUDAColorCvt_YUV2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	//qDebug("setSourceGPUResourceBuffer - 1");
	buffer_ptrs[index] = bits;

	if (index == 2)
	{
		int widths[3] = { imageBytesPerLine, imageBytesPerLine / 2, imageBytesPerLine / 2 };
		int heights[3] = { imageHeight, imageHeight / 2, imageHeight / 2 };
		int hoffset[3] = { 0, imageHeight, imageHeight * 5 / 4};
		
		//qDebug("setSourceGPUResourceBuffer - 3");
		if (imageBytesPerLine > 0 && imageHeight > 0)
		{
			cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;

			for (int i = 0; i < 3; i++)
			{
				int len = widths[i] * heights[i];
				cudaMemcpyToArrayAsync(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, hoffset[i], buffer_ptrs[i], len, cudaMemcpyHostToDevice, stream);
			}
		}
	}
}

void CUDAColorCvt_YUV2RGBA::runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	runColorCvt_YUV2RGBA_Kernel(outputSurf, inputTex, width, height, imageBytesPerLine);
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[width * height*4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, width *height *4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, width, height, QImage::Format_RGBA8888);
	img.save(QString("yuv2rgbacvt.png"));
	delete[] buffer;
#endif
}

bool CUDAColorCvt_YUV2RGBA::isColorConversionNeeded()
{
	return true;
}

//////////////

CUDAColorCvt_YUV4222RGBA::CUDAColorCvt_YUV4222RGBA(QObject *parent) : CUDAColorCvt_2RGBA(parent)
{
}

CUDAColorCvt_YUV4222RGBA::~CUDAColorCvt_YUV4222RGBA()
{

}

void CUDAColorCvt_YUV4222RGBA::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageBytesPerLine / 4, imageHeight, cudaArraySurfaceLoadStore);
	}
}


void CUDAColorCvt_YUV4222RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;

	int widthPixels = imageBytesPerLine / 4;

	if (imageBytesPerLine > 0 && imageHeight > 0)
	{
		cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;

		cudaMemcpyToArrayAsync(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, 0, bits, imageBytesPerLine * imageHeight, cudaMemcpyHostToDevice, stream);
	}
}

void CUDAColorCvt_YUV4222RGBA::runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	runColorCvt_YUV4222RGBA_Kernel(outputSurf, inputTex, width, height, imageBytesPerLine);
	
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[width * height * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, width *height * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, width, height, QImage::Format_RGBA8888);
	img.save(QString("yuv4222rgbacvt.png"));
	delete[] buffer;
#endif
}

bool CUDAColorCvt_YUV4222RGBA::isColorConversionNeeded()
{
	return true;
}

//////////////

CUDAColorCvt_YUVJ422P2RGBA::CUDAColorCvt_YUVJ422P2RGBA(QObject *parent) : CUDAColorCvt_2RGBA(parent)
{
}

CUDAColorCvt_YUVJ422P2RGBA::~CUDAColorCvt_YUVJ422P2RGBA()
{

}

void CUDAColorCvt_YUVJ422P2RGBA::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageBytesPerLine, imageHeight * 2, cudaArraySurfaceLoadStore);
	}

}

void CUDAColorCvt_YUVJ422P2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	//qDebug("setSourceGPUResourceBuffer - 1");
	buffer_ptrs[index] = bits;

	if (index == 2)
	{
		int widths[3] = { imageBytesPerLine, imageBytesPerLine / 2, imageBytesPerLine / 2 };
		int heights[3] = { imageHeight, imageHeight, imageHeight };
		int hoffset[3] = { 0, imageHeight, imageHeight * 3 / 2 };

		//qDebug("setSourceGPUResourceBuffer - 3");
		if (imageBytesPerLine > 0 && imageHeight > 0)
		{
			cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;
			for (int i = 0; i < 3; i++)
			{
				int len = widths[i] * heights[i];
				cudaMemcpyToArrayAsync(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, hoffset[i], buffer_ptrs[i], len, cudaMemcpyHostToDevice, stream);
			}
		}
	}
}

void CUDAColorCvt_YUVJ422P2RGBA::runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	runColorCvt_YUVJ422P2RGBA_Kernel(outputSurf, inputTex, width, height, imageBytesPerLine);
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[width * height * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, width *height * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, width, height, QImage::Format_RGBA8888);
	img.save(QString("yuv2rgbacvt.png"));
	delete[] buffer;
#endif
}

bool CUDAColorCvt_YUVJ422P2RGBA::isColorConversionNeeded()
{
	return true;
}

//////////////
CUDAColorCvt_BGR02RGBA::CUDAColorCvt_BGR02RGBA(QObject *parent) : CUDAColorCvt_2RGBA(parent)
{
}

CUDAColorCvt_BGR02RGBA::~CUDAColorCvt_BGR02RGBA()
{

}

void CUDAColorCvt_BGR02RGBA::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++) {
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageBytesPerLine, imageHeight, cudaArraySurfaceLoadStore);
	}
}


void CUDAColorCvt_BGR02RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;

	if (imageWidth > 0 && imageHeight > 0)
	{
		cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;
		cudaMemcpyToArrayAsync(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, 0, bits, imageBytesPerLine * imageHeight, cudaMemcpyHostToDevice, stream);
	}
}

void CUDAColorCvt_BGR02RGBA::runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	runColorCvt_BGR02RGBA_Kernel(outputSurf, inputTex, width, height, imageBytesPerLine);
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[width * height * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, width *height * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, width, height, QImage::Format_RGBA8888);
	img.save(QString("rgb2rgbacvt.png"));
	delete[] buffer;
#endif
}

bool CUDAColorCvt_BGR02RGBA::isColorConversionNeeded()
{
	return false;
}

//////////////
CUDAColorCvt_RGB2RGBA::CUDAColorCvt_RGB2RGBA(QObject *parent) : CUDAColorCvt_2RGBA(parent)
{
}

CUDAColorCvt_RGB2RGBA::~CUDAColorCvt_RGB2RGBA()
{

}

void CUDAColorCvt_RGB2RGBA::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageBytesPerLine, imageHeight, cudaArraySurfaceLoadStore);
	}
}


void CUDAColorCvt_RGB2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;

	if (imageWidth > 0 && imageHeight > 0)
	{
		cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;
		cudaMemcpyToArrayAsync(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, 0, bits, imageBytesPerLine * imageHeight, cudaMemcpyHostToDevice, stream);
	}
}

void CUDAColorCvt_RGB2RGBA::runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	runColorCvt_RGB2RGBA_Kernel(outputSurf, inputTex, width, height, imageBytesPerLine);
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[width * height * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, width *height * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, width, height, QImage::Format_RGBA8888);
	img.save(QString("rgb2rgbacvt.png"));
	delete[] buffer;
#endif
}

bool CUDAColorCvt_RGB2RGBA::isColorConversionNeeded()
{
	return false;
}

//////////////
CUDAColorCvt_RGBA2RGBA::CUDAColorCvt_RGBA2RGBA(QObject *parent) : CUDAColorCvt_2RGBA(parent)
{
}

CUDAColorCvt_RGBA2RGBA::~CUDAColorCvt_RGBA2RGBA()
{

}

void CUDAColorCvt_RGBA2RGBA::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageWidth, imageHeight, cudaArraySurfaceLoadStore);
	}
	
}

void CUDAColorCvt_RGBA2RGBA::setSourceGPUResourceBuffer(int index, const uchar* bits, bool textureChanged)
{
	if (index != 0)
		return;

	if (imageWidth > 0 && imageHeight > 0)
	{
		cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;
		cudaMemcpyToArray(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, 0, bits, imageBytesPerLine * imageHeight, cudaMemcpyHostToDevice);
	}
}

void CUDAColorCvt_RGBA2RGBA::runKernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int byteperline)
{
	cudaError err = cudaGetLastError();
	runColorCvt_RGBA2RGBA_Kernel(outputSurf, inputTex, width, height, imageBytesPerLine);

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[width * height * 4];
	err = cudaMemcpyFromArray(buffer, m_cudaTargetArrays[m_workingGPUResource], 0, 0, width *height* 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, width, height, QImage::Format_RGBA8888);
	img.save(QString("rgba2rgbacvt") + ".png");
	delete[] buffer;
#endif
}

bool CUDAColorCvt_RGBA2RGBA::isColorConversionNeeded()
{
	return false;
}

////

CUDAUniColorCvt::CUDAUniColorCvt() :GPUUniColorCvt()
{

}
CUDAUniColorCvt::~CUDAUniColorCvt()
{

}

bool CUDAUniColorCvt::DynRender(ImageBufferData& img, void* gl)
{
	bool initialized = false;
	if (mFormat == img.mFormat
		&& mConverter != NULL
		&& mConverter->ImageWidth() == img.mImageY.width
		&& mConverter->ImageHeight() == img.mImageY.height)
	{
		initialized = true;
	}

	if (!initialized)
	{	// reset
		Free();

		mFormat = img.mFormat;
		switch (img.mFormat)
		{
		case ImageBufferData::YUV420:
			mConverter = new CUDAColorCvt_YUV2RGBA();
			break;
		case ImageBufferData::RGB888:
			mConverter = new CUDAColorCvt_RGB2RGBA();
			break;
		case ImageBufferData::RGBA8888:
			mConverter = new CUDAColorCvt_RGBA2RGBA();
			break;
		}
		if (mConverter == NULL)
			return false;
		
		mConverter->initialize(img.mImageY.width, img.mImageY.height);
	}
	mConverter->render(img);
	return true;
}

// Alpha Texture
CUDAAlphaSource::CUDAAlphaSource(QObject *parent) : GPUAlphaSource(parent)
{
}

CUDAAlphaSource::~CUDAAlphaSource()
{

	if (m_initialized)
	{
		
		
		for (int i = 0; i < m_targetCount; i++){
			cudaFreeArray(m_cudaSrcArrays[i]);
			cudaDestroySurfaceObject(m_cudaSrcSurfaces[i]);
			cudaDestroyTextureObject(m_cudaSrcTextures[i]);

			cudaFreeArray(m_cudaTargetArrays[i]);
			cudaDestroySurfaceObject(m_cudaTargetSurfaces[i]);
			cudaDestroyTextureObject(m_cudaTargetTextures[i]);
		}
	}
}

void CUDAAlphaSource::initialize(int imgWidth, int imgHeight)
{

	imageWidth = 0;
	imageHeight = 0;
	imageBytesPerLine = 0;

	for (int i = 0; i < m_targetCount; i++)
	{
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaTargetArrays[i], &channelFormat, imageWidth, imageHeight, cudaArraySurfaceLoadStore);

		cudaResourceDesc    surfRes;
		memset(&surfRes, 0, sizeof(cudaResourceDesc));
		surfRes.resType = cudaResourceTypeArray;
		surfRes.res.array.array = m_cudaTargetArrays[i];
		cudaCreateSurfaceObject(&m_cudaTargetSurfaces[i], &surfRes);

		cudaTextureDesc             texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));

		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;

		texDescr.addressMode[0] = cudaAddressModeClamp;
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;

		texDescr.readMode = cudaReadModeNormalizedFloat;

		cudaCreateTextureObject(&m_cudaTargetTextures[i], &surfRes, &texDescr, NULL);
	}
	m_workingGPUResource = 0;
	m_targetArrayIndex = 0;

	m_initialized = true;
}


void CUDAAlphaSource::render(QImage& img)
{

	int width = img.width();
	int height = img.height();

	if (width == 0 || height == 0) return;

	bool blTextureFormatChanged = false;
	if (imageWidth != width || imageHeight != height)
	{
		imageWidth = width;
		imageHeight = height;
		imageBytesPerLine = width;

		createSourceGPUResource();

		blTextureFormatChanged = true;
	}

	

	const unsigned char * buffer = img.bits();

	m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
	// host to device buffer uploading.
	setSourceGPUResourceBuffer(buffer, blTextureFormatChanged);

	// Don't comment below code block absolutely.
	if (blTextureFormatChanged)
	{	// It's very important to avoid empty-rendering.
		m_targetArrayIndex = (m_targetArrayIndex + 1) % m_targetCount;
		// host to device buffer uploading.
		setSourceGPUResourceBuffer(buffer, false);

		for (int i = 0; i < m_targetCount; i++){
			cudaResourceDesc    surfRes;
			memset(&surfRes, 0, sizeof(cudaResourceDesc));
			surfRes.resType = cudaResourceTypeArray;
			surfRes.res.array.array = m_cudaSrcArrays[i];
			cudaCreateSurfaceObject(&m_cudaSrcSurfaces[i], &surfRes);

			cudaTextureDesc             texDescr;
			memset(&texDescr, 0, sizeof(cudaTextureDesc));

			texDescr.normalizedCoords = 1;
			texDescr.filterMode = cudaFilterModeLinear;

			texDescr.addressMode[0] = cudaAddressModeClamp;
			texDescr.addressMode[1] = cudaAddressModeClamp;
			texDescr.addressMode[2] = cudaAddressModeClamp;

			texDescr.readMode = cudaReadModeNormalizedFloat;

			cudaCreateTextureObject(&m_cudaSrcTextures[i], &surfRes, &texDescr, NULL);
		}
		
	}
	GLfloat vertices[] = {
		-imageWidth / 2, -imageHeight / 2,
		-imageWidth / 2, imageHeight / 2,
		imageWidth / 2, imageHeight / 2,
		imageWidth / 2, -imageHeight / 2
	};

	GLfloat texCoords[] = {
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 1.0f,
		1.0f, 0.0f,
	};
	// source image(from ffmpeg) is y-inverted image, so convert to opengl-coord.
	refineTexCoordsForOutput(texCoords, 4);

	//insert cuda kernel.

	m_workingGPUResource = (m_workingGPUResource + 1) % m_targetCount;

}


const int CUDAAlphaSource::getWidth()
{
	return imageWidth;
}

const int CUDAAlphaSource::getHeight()
{
	return imageHeight;
}

void CUDAAlphaSource::createSourceGPUResource()
{
	for (int i = 0; i < m_targetCount; i++){
		cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
		cudaMallocArray(&m_cudaSrcArrays[i], &channelFormat, imageBytesPerLine, imageHeight * 3 / 2, cudaArraySurfaceLoadStore);
	}
	
}

void CUDAAlphaSource::setSourceGPUResourceBuffer(const uchar* bits, bool textureChanged)
{

	buffer_ptr = bits;

	if (getWidth() > 0 && getHeight() > 0)
	{
		cudaStream_t stream = (m_targetArrayIndex + 1) % m_targetCount == 0 ? g_CurStream : g_NextStream;
		cudaMemcpyToArrayAsync(m_cudaSrcArrays[(m_targetArrayIndex + 1) % m_targetCount], 0, 0, buffer_ptr, imageBytesPerLine * getHeight(), cudaMemcpyHostToDevice, stream);
	}
}

// Nodal Input
CUDANodalInput::CUDANodalInput() :GPUNodalInput()
{
}

CUDANodalInput::~CUDANodalInput()
{
}

void CUDANodalInput::createColorCvt(void* gl, bool liveMode, QString weightFilename)
{
	if (liveMode)
	{
		colorCvt = new CUDAColorCvt_YUV4222RGBA();
	}
	else
	{
		colorCvt = new CUDAColorCvt_YUV2RGBA();
	}
	colorCvt->setGL((QOpenGLFunctions *)gl);

	if (weightFilename != "")
	{
		weightMap = new CUDAAlphaSource();
		weightMap->setGL((QOpenGLFunctions *)gl);
		weightImage.load(weightFilename);
	}
}

void CUDANodalInput::initialize(int imgWidth, int imgHeight)
{
	colorCvt->initialize(imgWidth, imgHeight);
	if (weightMap)
	{
		weightMap->initialize(weightImage.width(), weightImage.height());
	}
	weightRenderCount = 0;
}


void CUDANodalInput::render(ImageBufferData& img)
{
	colorCvt->render(img);
	if (weightMap && weightRenderCount < 1)
	{
		weightMap->render(weightImage);
		weightRenderCount++;
	}
}

#endif //USE_CUDA
