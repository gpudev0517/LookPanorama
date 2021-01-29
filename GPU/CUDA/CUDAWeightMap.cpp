
#ifdef USE_CUDA
#include "CUDAWeightMap.h"
#include "define.h"
#include "common.h"

extern "C" void runCameraWeightMap_Kernel(cudaSurfaceObject_t outputSurf, int width, int height, float blendFalloff, 
	float fisheyeLensRadiusRatio1, float fisheyeLensRadiusRatio2, float xrad1, float xrad2, float yrad1, float yrad2, float blendCurveStart, int lens);

extern "C" void runPanoramaWeightMap_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t weightTex, cudaTextureObject_t deltaWeightTex, int width, int height,
	int lens, float cx, float cy, float offset_x, float offset_y, float k1, float k2, float k3, float FoV, float FoVY, float *cP);
//Camera Weight Map
CUDACameraWeightMap::CUDACameraWeightMap(QObject *parent) : GPUCameraWeightMap(parent)
{
}

CUDACameraWeightMap::~CUDACameraWeightMap()
{
}

void CUDACameraWeightMap::initialize(int imageWidth, int imageHeight)
{
	this->imageWidth = imageWidth;
	this->imageHeight = imageHeight;

	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&m_cudaTargetArray, &channelFormat, imageWidth, imageHeight, cudaArraySurfaceLoadStore);


	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);

	m_initialized = true;
}

void CUDACameraWeightMap::updateCameraParams()
{
}

void CUDACameraWeightMap::render(int camID)
{
	float blendingFalloff = 0.0f;
	if (camInput.m_cameraParams.isFisheye())
		blendingFalloff = 1.0f;
	else
		blendingFalloff = 0.5f;
	runCameraWeightMap_Kernel(m_cudaTargetSurface, imageWidth, imageHeight, blendingFalloff, camInput.m_cameraParams.m_xrad1, camInput.m_cameraParams.m_yrad1,
		camInput.m_cameraParams.m_xrad1, camInput.m_cameraParams.m_xrad2, camInput.m_cameraParams.m_yrad1, camInput.m_cameraParams.m_yrad2, 0.4f, camInput.m_cameraParams.m_lensType);
	
#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[imageWidth * imageHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, imageWidth *imageHeight, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, imageWidth, imageHeight, QImage::Format_Grayscale8);
	img.save("CameraWeightmap.png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
}


//Panorama Weight Map


CUDAPanoramaWeightMap::CUDAPanoramaWeightMap(QObject *parent, bool isYUV) : GPUPanoramaWeightMap(parent)
{
	m_devMat = NULL;
}

CUDAPanoramaWeightMap::~CUDAPanoramaWeightMap()
{
	if (m_initialized)
	{
		cudaFree(m_devMat);
	}
}

void CUDAPanoramaWeightMap::initialize(int xres, int yres, int panoWidth, int panoHeight)
{
	panoramaWidth = panoWidth;
	panoramaHeight = panoHeight;

	inputWidth = xres;
	inputHeight = yres;


	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&m_cudaTargetArray, &channelFormat, panoramaWidth, panoramaHeight, cudaArraySurfaceLoadStore);


	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeNormalizedFloat;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);

	m_initialized = true;
}

void CUDAPanoramaWeightMap::updateCameraParams()
{
	if (!m_devMat){
		cudaMalloc(&m_devMat, 9 * sizeof(float));
	}
	mat3 m = getCameraViewMatrix(camInput.m_cameraParams.m_yaw, camInput.m_cameraParams.m_pitch, camInput.m_cameraParams.m_roll);
	cudaMemcpy(m_devMat, m.mat_array, 9 * sizeof(float), cudaMemcpyHostToDevice);
}

void CUDAPanoramaWeightMap::render(unsigned int weightTextureId, unsigned int deltaWeightTexId, int camID)
{
	float nwidth = camInput.xres;
	float nheight = camInput.yres;
	

	runPanoramaWeightMap_Kernel(m_cudaTargetSurface, weightTextureId, deltaWeightTexId, panoramaWidth, panoramaHeight, camInput.m_cameraParams.m_lensType, nwidth / 2.f, nheight / 2.f,
		camInput.m_cameraParams.m_offset_x, camInput.m_cameraParams.m_offset_y, camInput.m_cameraParams.m_k1, camInput.m_cameraParams.m_k2, camInput.m_cameraParams.m_k3,
		camInput.m_cameraParams.m_fov, camInput.m_cameraParams.m_fovy, m_devMat);
 	

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_Grayscale8);
	img.save(QString("PanoramaWeightmap_") + QString::number(camID) + ".png");
	delete[] buffer;
#endif
}


//Delta Weight Map

CUDADeltaWeightMap::CUDADeltaWeightMap(QObject *parent, bool isYUV) : GPUDeltaWeightMap(parent, isYUV)
{
}

CUDADeltaWeightMap::~CUDADeltaWeightMap()
{
	if (m_initialized)
	{
		cudaGraphicsUnregisterResource(m_cudaFboTextureId);
		m_cudaTargetArray = NULL;
	}
}

void CUDADeltaWeightMap::initialize(int xres, int yres, int panoWidth, int panoHeight)
{
	GPUDeltaWeightMap::initialize(xres, yres, panoWidth, panoHeight);

	cudaGraphicsGLRegisterImage(&m_cudaFboTextureId, m_fboTextureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
	refreshCUDAArray();

	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);

}
void CUDADeltaWeightMap::resetMap()
{
	
	GPUDeltaWeightMap::resetMap();

	if (m_initialized){
		refreshCUDAArray();
	}
	

#if 0
	if (m_initialized){
		GLubyte *buffer = new GLubyte[camWidth * camHeight * 4];
		cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, camWidth *camHeight * 4, cudaMemcpyDeviceToHost);
		QImage img((uchar*)buffer, camWidth, camHeight, QImage::Format_RGBA8888);
		img.save(QString("DeltaWeightMap_") + ".png");
		delete[] buffer;
		if (err != cudaSuccess)
		{
			int a = 0;
			a++;
		}
	}
#endif
}

void CUDADeltaWeightMap::refreshCUDAArray()
{
	cudaGraphicsMapResources(1, &m_cudaFboTextureId, 0);
	cudaGraphicsSubResourceGetMappedArray(&m_cudaTargetArray, m_cudaFboTextureId, 0, 0);
	cudaGraphicsUnmapResources(1, &m_cudaFboTextureId, 0);
}

void CUDADeltaWeightMap::unRegisterCUDAArray()
{
	cudaGraphicsUnregisterResource(m_cudaFboTextureId);
	cudaDestroySurfaceObject(m_cudaTargetSurface);
	cudaDestroyTextureObject(m_cudaTargetTexture);
}

void CUDADeltaWeightMap::registerCUDAArray()
{
	cudaGraphicsGLRegisterImage(&m_cudaFboTextureId, m_fboTextureId, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly);
	refreshCUDAArray();

	cudaResourceDesc    surfRes;
	memset(&surfRes, 0, sizeof(cudaResourceDesc));
	surfRes.resType = cudaResourceTypeArray;
	surfRes.res.array.array = m_cudaTargetArray;
	cudaCreateSurfaceObject(&m_cudaTargetSurface, &surfRes);

	cudaTextureDesc             texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));

	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;

	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;

	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&m_cudaTargetTexture, &surfRes, &texDescr, NULL);
}

GPUResourceHandle CUDADeltaWeightMap::getTargetBuffer()
{
	return m_fboId;
}

#endif //USE_CUDA