#ifdef USE_CUDA
#include "CUDAPanoramaPostProcessing.h"
#include "define.h"

extern "C" void runPanoramaPostProcessing_Kernel(cudaSurfaceObject_t outputSurf, cudaTextureObject_t inputTex, cudaSurfaceObject_t lutInpuSurf, cudaTextureObject_t seamInputTex,
	int width, int height, float3 ctLightColor, float *placeMat, bool seamOn);

CUDAPanoramaPostProcessing::CUDAPanoramaPostProcessing(QObject *parent) : GPUPanoramaPostProcessing(parent)
{
	cudaMalloc(&m_devPlaceMat, 9 * sizeof(float));
}

CUDAPanoramaPostProcessing::~CUDAPanoramaPostProcessing()
{
	cudaFree(m_devPlaceMat);
	cudaFreeArray(m_lutArray);
	cudaDestroySurfaceObject(m_lutSurface);
}

void CUDAPanoramaPostProcessing::initialize(int panoWidth, int panoHeight)
{
	this->panoramaWidth = panoWidth;
	this->panoramaHeight = panoHeight;

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


	cudaChannelFormatDesc channelFormat1 = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
	cudaMallocArray(&m_lutArray, &channelFormat1, 256, 1, cudaArraySurfaceLoadStore);

	cudaResourceDesc    surfRes1;
	memset(&surfRes1, 0, sizeof(cudaResourceDesc));
	surfRes1.resType = cudaResourceTypeArray;
	surfRes1.res.array.array = m_lutArray;
	cudaCreateSurfaceObject(&m_lutSurface, &surfRes1);


	m_initialized = true;
}

void CUDAPanoramaPostProcessing::setLutData(QVariantList *vList){

	float splitUnit = 255.f / (float)(LUT_COUNT-1);

	GLubyte *ptr = new GLubyte[256 * 4];
	for (int k = 0; k < 4; k++){
		int s_j = 0;
		for (int i = 0; i < 256; i++){
			for (int j = 0; j < vList[k].size() - 1; j++){
				if (i >= j * splitUnit && i < (j + 1) * splitUnit){
					s_j = j;
					break;
				}
			}
			float alpha = (i - s_j * splitUnit) / splitUnit;
			float sample = (1.0f - alpha) * vList[k][s_j].toFloat() + alpha * vList[k][s_j + 1].toFloat();
			ptr[4 * i + k] = fmin(sample * 255.f, 255.f);
		}
	}

	cudaMemcpyToArray(m_lutArray, 0, 0, ptr, 256 * 4 * sizeof(GLubyte), cudaMemcpyHostToDevice);
	delete[] ptr;
}

void CUDAPanoramaPostProcessing::updateGlobalParams(float yaw, float pitch, float roll)
{
	mat3 m = mat3_id, invM = mat3_id;
	vec3 u(roll * sd_to_rad, pitch * sd_to_rad, yaw * sd_to_rad);
	m.set_rot_zxy(u);
	invert(invM, m);

	cudaMemcpy(m_devPlaceMat, invM.mat_array, sizeof(float)* 9, cudaMemcpyHostToDevice);
}

void CUDAPanoramaPostProcessing::render(GLuint panoTextureId, vec3 ctLightColor,
	float yaw, float pitch, float roll,
	int seamTextureId)
{

	float width = getWidth();
	float height = getHeight();

	runPanoramaPostProcessing_Kernel(m_cudaTargetSurface, panoTextureId, m_lutSurface, seamTextureId, width, height, make_float3(ctLightColor.x, ctLightColor.y, ctLightColor.z), m_devPlaceMat, (seamTextureId != -1 ? true : false));
	//runPanoramaColorCorrection_Kernel(m_cudaTargetSurface, panoTextureId, m_lutSurface, width, height, make_float3(ctLightColor.x, ctLightColor.y, ctLightColor.z));
	

#if 0
	cudaDeviceSynchronize();
	GLubyte *buffer = new GLubyte[panoramaWidth * panoramaHeight * 4];
	cudaError err = cudaMemcpyFromArray(buffer, m_cudaTargetArray, 0, 0, panoramaWidth *panoramaHeight * 4, cudaMemcpyDeviceToHost);
	QImage img((uchar*)buffer, panoramaWidth, panoramaHeight, QImage::Format_RGBA8888);
	img.save(QString("PanoramaColorCorrection_") + QString::number(panoTextureId) + ".png");
	delete[] buffer;
	if (err != cudaSuccess)
	{
		int a = 0;
		a++;
	}
#endif
}

#endif //USE_CUDA