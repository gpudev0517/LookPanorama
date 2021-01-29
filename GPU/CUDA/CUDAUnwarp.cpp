#ifdef USE_CUDA

#include "CUDAUnwarp.h"

extern "C" void runUnwarp_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, int width, int height, int renderMode, int lens, float imageWidth, float imageHeight,
	float offset_x, float offset_y, float k1, float k2, float k3, float FoV, float FoVY, float *cP);
CUDAUnwarp::CUDAUnwarp(QObject *parent, bool isYUV) : GPUUnwarp(parent)
{
	cudaMalloc(&m_devMat, sizeof(float) * 9);
}

CUDAUnwarp::~CUDAUnwarp()
{
	cudaFree(m_devMat);
}


void CUDAUnwarp::initialize(int id, int xres, int yres, int panoWidth, int panoHeight)
{
	camID = id;
	panoramaWidth = panoWidth;
	panoramaHeight = panoHeight;

	inputWidth = xres;
	inputHeight = yres;

	cudaChannelFormatDesc channelFormat = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
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

void CUDAUnwarp::updateCameraParams()
{
	mat3 m = getCameraViewMatrix(camInput.m_cameraParams.m_yaw, camInput.m_cameraParams.m_pitch, camInput.m_cameraParams.m_roll);
	cudaMemcpy(m_devMat, m.mat_array, 9 * sizeof(float), cudaMemcpyHostToDevice);
}

void CUDAUnwarp::render(GPUResourceHandle inputSurf, RenderMode renderMode)
{
	runUnwarp_Kernel(m_cudaTargetSurface, inputSurf, panoramaWidth, panoramaHeight, renderMode, camInput.m_cameraParams.m_lensType, camInput.xres, camInput.yres,
		camInput.m_cameraParams.m_offset_x, camInput.m_cameraParams.m_offset_y, camInput.m_cameraParams.m_k1, camInput.m_cameraParams.m_k2, camInput.m_cameraParams.m_k3,
		camInput.m_cameraParams.m_fov, camInput.m_cameraParams.m_fovy, m_devMat);

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_RGBA8888);
	img.save(QString("Unwarp_") + QString::number(camID) + ".png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif

	
}

#endif //USE_CUDA