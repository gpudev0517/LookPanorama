

#ifndef _CUDACOMMON_CUH_
#define _CUDACOMMON_CUH_

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>
#include <vector_types.h>
#include <driver_functions.h>


#define CAM_BLOCK_NUM 32
#define PANO_BLOCK_NUM 32

#define M_PI        3.14159265358979323846f
#define M_PI_2      1.57079632679489661923f
#define M_PI_4      0.785398163397448309616f

#define   LensModel_standard      0
#define   LensModel_fisheye       1
#define   LensModel_opencv_standard 3
#define   LensModel_opencv_fisheye 4

#define RenderMode_Color 0
#define RenderMode_Weight 1

extern cudaStream_t g_CurStream;
extern cudaStream_t g_NextStream;


__device__ __inline__ uchar4 to_uchar4(float4 vec)
{
	return make_uchar4((unsigned char)vec.x, (unsigned char)vec.y, (unsigned char)vec.z, (unsigned char)vec.w);
}

__device__ __inline__ float curve(float s)
{
	return s;
}

__device__ __inline__ uchar4 mix(uchar4 u, uchar4 v, float alpha)
{
	uchar4 ret;
	ret.x = __fadd_rn(__fmul_rn(1.f - alpha, u.x), __fmul_rn(alpha, v.x));
	ret.y = __fadd_rn( __fmul_rn(1.f - alpha, u.y), __fmul_rn(alpha, v.y));
	ret.z = __fadd_rn(__fmul_rn(1.f - alpha, u.z), __fmul_rn(alpha, v.z));
	ret.w = __fadd_rn(__fmul_rn(1.f - alpha, u.w), __fmul_rn(alpha, v.w));
	return ret;
}
__device__ __inline__ float4 mix(float4 u, float4 v, float alpha)
{
	float4 ret;
	ret.x = __fadd_rn(__fmul_rn(1.f - alpha, u.x), __fmul_rn(alpha, v.x));
	ret.y = __fadd_rn(__fmul_rn(1.f - alpha, u.y), __fmul_rn(alpha, v.y));
	ret.z = __fadd_rn(__fmul_rn(1.f - alpha, u.z), __fmul_rn(alpha, v.z));
	ret.w = __fadd_rn(__fmul_rn(1.f - alpha, u.w), __fmul_rn(alpha, v.w));
	return ret;
}


__device__ __inline__ float clamp(float value, float low, float high)
{
	return value < low ? low : (value > high ? high : value);
}

__device__ __inline__ float3 mult(float *M, float3 v){
	float3 u;
	u.x = __fmul_rn(M[0], v.x) + __fmul_rn(M[3], v.y) + __fmul_rn(M[6], v.z);
	u.y = __fmul_rn(M[1], v.x) + __fmul_rn(M[4], v.y) + __fmul_rn(M[7] , v.z);
	u.z = __fmul_rn(M[2], v.x) + __fmul_rn(M[5], v.y) + __fmul_rn(M[8] , v.z);
	return u;
}

__device__ __inline__ float dot(float3 u, float3 v){
	return __fmul_rn(u.x, v.x) + __fmul_rn(u.y, v.y) + __fmul_rn(u.z, v.z);
}

__device__ __inline__ float getRadialDistance2(float2 imageUV, int width, int height, float blendingFalloff,
	float fisheyeLensRadiusRatio1, float fisheyeLensRadiusRatio2, float xrad1, float xrad2, float yrad1, float yrad2, float blendCurveStart, int lens)
{
	float dist2 = 0.0f;

	if (lens == LensModel_fisheye || lens == LensModel_opencv_fisheye)
	{
		float xdist = __fdividef(( __fdividef(width , 2.f) - imageUV.x) , __fdividef(width , 2.f));
		if (__fmul_rn(imageUV.x , 2.f) < width)
			xdist = __fdividef(xdist, xrad1);
		else
			xdist = __fdividef(xdist , xrad2);
		xdist = __fmul_rn(xdist , xdist);
		//xdist = xdist * xdist;
		 
		float ydist = __fdividef((__fdividef(height , 2.f) - imageUV.y) , __fdividef(height , 2.f));
		if (__fmul_rn(imageUV.y, 2) > height)
			ydist = __fdividef(ydist , yrad1);
		else
			ydist = __fdividef(ydist , yrad2);
		ydist = __fmul_rn(ydist , ydist);
		//ydist = ydist * ydist;

		dist2 = xdist + ydist;
	}
	else if (lens == LensModel_standard || lens == LensModel_opencv_standard)
	{
		float xdist = __fdividef((__fdividef(width , 2.f) - imageUV.x) , (__fdividef(width , 2.f) * fisheyeLensRadiusRatio1));
		float ydist = __fdividef((__fdividef(height , 2.f) - imageUV.y) , (__fdividef(height , 2.f) * fisheyeLensRadiusRatio2));
		if (xdist < 0) xdist = -xdist;
		if (ydist < 0) ydist = -ydist;

		dist2 = 1;
		if (xdist >= ydist)
		{
			dist2 = xdist;
		}
		else
		{
			dist2 = ydist;
		}
	}
	dist2 = __powf(dist2, __fdividef(blendingFalloff, 2.f));

	// convert [0, start, end=1.0f] to [0, 0, 1]
	float start = blendCurveStart;
	float end = 1.0f;
	if (start >= end) start = 0.0f;
	dist2 = __saturatef((dist2 - start) * __fdividef(1.0f , (end - start))); 

	return dist2;
}

__device__ __inline__ void sphericalToCartesian(float theta, float phi, float3 &cartesian) {
	float sinphi, cosphi, sintheta, costheta;
	__sincosf(phi, &sinphi, &cosphi);
	__sincosf(theta, &sintheta, &costheta);
	cartesian.y = sinphi;
	cartesian.x = __fmul_rn(cosphi, sintheta);
	cartesian.z = __fmul_rn(cosphi, costheta);
}


__device__ __inline__ float cartesianTospherical(float3 cartesian, float &theta, float &phi) {
	float result = 1.0f;
	phi = asinf(cartesian.y);
	float cosphi = __powf(1.0f - __fmul_rn(cartesian.y, cartesian.y), 0.5f);
	if (cosphi == 0.0f)
	{
		theta = 0.0f;
		return 0.0f;
	}
	else
	{
		theta = atan2f(cartesian.x, cartesian.z);
		return 1.0f;
	}
}

// return float instead of bool because bool / int do not work on MacBookPro 10.6
__device__ __inline__ void XYnToThetaPhi(float x_n, float y_n, float &theta, float &phi)
{
	theta = __fmul_rn(__fmaf_rn(2.0f, x_n, -1.0f) , M_PI);
	phi = __fmaf_rn(M_PI , y_n , -M_PI_2);
}

__device__ __inline__ float ThetaPhiToXYn(float theta, float phi, float &x_n, float &y_n)
{
	float result = 1.0f;
	x_n = __fdividef((__fdividef(theta,  M_PI) + 1.f), 2.f);
	y_n = __fdividef((phi + M_PI_2), M_PI);
	return result;
}

__device__ __inline__ void XYnToDstXYn(float2 xyn, float2 &dstXyn, float* placeMat) {
	float theta, phi;
	float3 cartesian;

	XYnToThetaPhi(xyn.x, xyn.y, theta, phi);
	sphericalToCartesian(theta, phi, cartesian);
	float3 U = mult(placeMat, cartesian);
	cartesianTospherical(U, theta, phi);
	float x, y;
	ThetaPhiToXYn(theta, phi, x, y);
	dstXyn.x = x;
	dstXyn.y = y;
}

// return float instead of bool because bool / int do not work on MacBookPro 10.6
__device__ __inline__ float toLocal(float3 cartesian, float scale, float2 &camera, int lens, float cx, float cy, float offset_x, float offset_y, float k1, float k2, float k3, float FoV, float FoVY, float *cP)
{
	float imageWidth = __fmul_rn(cx , 2.f);
	float imageHeight = __fmul_rn(cy , 2.f);
	float3 U = mult(cP, cartesian);

	float alpha = 1.0f;

	bool opencvLens = (lens >= LensModel_opencv_standard) ? true : false;
	if (opencvLens)
	{
		if (lens == LensModel_opencv_standard)
		{
			float focal = __fdividef(__fdividef(imageWidth, 2.f), __tanf(__fdividef(__fmul_rn(FoV, M_PI), 360.f)));
			//float focal = imageWidth / 2.f / __tanf(FoV / 2.f * M_PI / 180.0f);
			float a = __fdividef(U.x, U.z);
			float b = __fdividef(U.y, U.z);
			//float a = U.x / U.z;
			//float b = U.y / U.z;
			if (U.z <= 0.001f)
			{
				alpha = 0.0f;
			}
			else
			{
				float r = __fsqrt_rn(__fmul_rn(a, a) + __fmul_rn(b,b));
				float r2 = __fmul_rn(r,r);
				//float dist1 = (1.f + (k1 + (k2 + k3*r2)*r2)*r2);
				float dist1 = __fmaf_rn(__fmaf_rn(__fmaf_rn(k3, r2, k2), r2, k1), r2, 1.f);

				if (dist1 >= 0.7)
				{
					float x_ = __fmul_rn(a, dist1);
					float y_ = __fmul_rn(b, dist1);

					float x = __fmaf_rn(focal, x_, cx + offset_x);
					float y = __fmaf_rn(focal, y_, cy + offset_y);

					camera.x = x;
					camera.y = y;

					if ((x < 0.0) || (x >= imageWidth) || (y < 0.0) || (y >= imageHeight))
						alpha = 0.0f;
				}
				else
				{
					alpha = 0.0f;
				}
			}
		}
		else if (lens == LensModel_opencv_fisheye)
		{
			float focalX = __fdividef(__fdividef(imageWidth, 2.f), __tanf(__fdividef(__fmul_rn(FoV, M_PI), 360.0f)));
			float focalY = __fdividef(__fdividef(imageWidth, 2.f), __tanf(__fdividef(__fmul_rn(FoVY, M_PI), 180.0f)));
			//float focalX = imageWidth / 2.f / __tanf(FoV / 2.f * M_PI / 180.0f);
			//float focalY = imageWidth / 2.f / __tanf(FoVY / 2.f * M_PI / 180.0f);
			float a = __fdividef(U.x, U.z);
			float b = __fdividef(U.y, U.z);
			/*float a = U.x / U.z;
			float b = U.y / U.z;*/
			if (U.z <= 0.001f)
			{
				alpha = 0.0f;
			}
			else
			{
				float r = __fsqrt_rn(__fmul_rn(a, a) + __fmul_rn(b,b));
				float theta = atanf(r);
				float theta2 = __fmul_rn(theta,theta);
				//float theta_distortion = theta * (1.f + (k1 + (k2 + k3*theta2)*theta2)*theta2);
				float theta_distortion = __fmul_rn(__fmaf_rn(__fmaf_rn(__fmaf_rn(k3, theta2, k2), theta2, k1), theta2, 1.f), theta);
				float x_ = __fmul_rn(__fdividef(theta_distortion , r), a);
				float y_ = __fmul_rn(__fdividef(theta_distortion, r) , b);

				float x = __fmaf_rn(focalX , x_ , cx + offset_x);
				float y = __fmaf_rn(focalY , y_ , cy + offset_y);

				camera.x = x;
				camera.y = y;

				if ((x < 0.0f) || (x >= imageWidth) || (y < 0.0f) || (y >= imageHeight))
					alpha = 0.0f;
			}
		}
	}
	else
	{
		float theta = acosf(__fdividef(U.z, __fsqrt_rn(__fmul_rn(U.x, U.x) + __fmul_rn(U.y, U.y) + __fmul_rn(U.z , U.z))));
		//float theta = acosf(U.z / sqrtf(U.x * U.x + U.y * U.y + U.z * U.z));

		float x_c = 0.f;
		float y_c = 0.f;
		float r = 0.f;
		float fisheye_radius = imageHeight;
		float fovHalfInRad = __fdividef(__fmul_rn(FoV , M_PI), 360.f);
		//float fovHalfInRad = FoV / 180.f * M_PI / 2.f;
		if (lens == LensModel_fisheye)
		{
			// for equidistant
			float f = __fdividef(__fdividef(imageWidth , 2.f ),  fovHalfInRad);
			r = __fmul_rn(f , theta);

			// fisheye equisolid
			//float f = imageWidth / 4 / sin(FoV/180*M_PI/4);
			//float r = 2 * f * sin(theta/2);

			// fisheye stereographic
			//float f = imageWidth / 4 / tan(FoV/180*M_PI/4);
			//float r = 2 * f * tan(theta/2);

			// orthogonal
			//float f = imageWidth/2 / sin(FoV/180*M_PI / 2);
			//float r = f * sin(theta);
		}
		else if (lens == LensModel_standard)
		{
			// Standard
			float f = __fdividef( __fdividef(imageWidth , 2.f ), __tanf(fovHalfInRad));
			r =__fmul_rn( f , __tanf(theta));
		}

		float r0 = __fdividef(fminf(imageWidth, imageHeight), 2.0f);
		float asp = __fdividef(fmaxf(imageWidth, imageHeight) , fminf(imageWidth, imageHeight));
		float rt = __fdividef(r , r0);
		if (lens == LensModel_fisheye)
		{
			rt = clamp(rt, 0.f, asp);
		}
		else if (lens == LensModel_standard)
		{
			if (rt < 0.f)
			{
				alpha = 0.0f;
				rt = 0.f;
			}
		}
		//float distScale = ((k1*rt + k2)*rt + k3)*rt + (1.f - k1 - k2 - k3);
		float distScale = __fmaf_rn(__fmaf_rn(__fmaf_rn(k1, rt, k2), rt, k3), rt, (1.f - k1 - k2 - k3));
		if (distScale < 0.1f)
			distScale = 0.1f;
		float rc = __fmul_rn(r, distScale);

		float xoff = offset_x;
		float yoff = -offset_y;

		float r2 = __fsqrt_rn(__fmul_rn(U.x , U.x) + __fmul_rn(U.y , U.y));
		float dx = U.x;
		float dy = U.y;
		if (r2 != 0.f)
		{
			dx = __fdividef(dx, r2);
			dy = __fdividef(dy, r2);
		}
		float Vx = __fmul_rn(rc , dx); // rc * cos(q);
		float Vy = __fmul_rn(rc , dy); // rc * sin(q);

		x_c = Vx + xoff;
		y_c = Vy + yoff;

		x_c += cx;
		y_c += cy;

		camera.x = __fmul_rn(x_c, scale);
		camera.y = __fmul_rn(y_c, scale);
		if (rt == 0.0f)
			alpha = 0.0f;

		if ((x_c < 0.0f) || (x_c >= imageWidth) || (y_c < 0.0f) || (y_c >= imageHeight))
			alpha = 0.0f;
	}


	return alpha;
}

__device__ __inline__ float XYnToLocal(float2 xyn, float scale, float2 &camera, int lens, float cx, float cy, float offset_x, float offset_y, float k1, float k2, float k3, float FoV, float FoVY, float *cP)
{
	float theta, phi;
	float3 cartesian;

	XYnToThetaPhi(xyn.x, xyn.y, theta, phi);
	sphericalToCartesian(theta, phi, cartesian);

	// cartesian to camera
	return toLocal(cartesian, scale, camera, lens, cx, cy, offset_x, offset_y, k1, k2, k3, FoV, FoVY, cP);
}

__device__ __inline__ bool UVToBannerCoord(float2 uv, float2 &xy, float *bannerPaiPlane, float bannerPaiZdotOrg, float *bannerHomography)
{
	float theta, phi;
	XYnToThetaPhi(uv.x, uv.y, theta, phi);

	float3 v;
	sphericalToCartesian(theta, phi, v);
	
	float3 bannerPaiPlane_2;
	bannerPaiPlane_2.x = bannerPaiPlane[8];
	bannerPaiPlane_2.y = bannerPaiPlane[9];
	bannerPaiPlane_2.z = bannerPaiPlane[10];
	float t = __fdividef(bannerPaiZdotOrg , dot(v, bannerPaiPlane_2));
	if (t <= 0.0f)
		return false;
	else
	{
		float3 vp;
		vp.x = __fmul_rn(t , v.x);
		vp.y = __fmul_rn(t , v.y);
		vp.z = __fmul_rn(t , v.z);
		vp.x -= bannerPaiPlane[12]; vp.y -= bannerPaiPlane[13]; vp.z -= bannerPaiPlane[14];
		//float3 bannerPaiPlane_0 = make_float3(bannerPaiPlane[0], bannerPaiPlane[1], bannerPaiPlane[2]);
		float xp = __fmul_rn(vp.x, bannerPaiPlane[0]) + __fmul_rn(vp.y, bannerPaiPlane[1]) + __fmul_rn(vp.z, bannerPaiPlane[2]);//dot(vp, bannerPaiPlane_0);
		//float3 bannerPaiPlane_1 = make_float3(bannerPaiPlane[4], bannerPaiPlane[5], bannerPaiPlane[6]);
		float yp = __fmul_rn(vp.x, bannerPaiPlane[4]) + __fmul_rn(vp.y, bannerPaiPlane[5]) + __fmul_rn(vp.z, bannerPaiPlane[6]);//dot(vp, bannerPaiPlane_1);

		float3 vt = mult(bannerHomography, make_float3(xp, yp, 1.0));
		vt.x = __fdividef(vt.x , vt.z); vt.y = __fdividef(vt.y , vt.z); vt.z = __fdividef(vt.z, vt.z);
		xy.x = vt.x; xy.y = vt.y;
		if (0.f <= vt.x && vt.x <= 1.f && 0.f <= vt.y && vt.y <= 1.f)
			return true;
		else
			return false;
	}
}

__device__ __inline__ uchar3 calcLut(cudaSurfaceObject_t lutInpuSurf, uchar4 color)
{
	uchar4 input;
	surf2Dread(&input, lutInpuSurf, color.x * sizeof(uchar4), 0);
	float r_ = __fdividef(input.y, 255.f);
	surf2Dread(&input, lutInpuSurf, color.y * sizeof(uchar4), 0);
	float g_ = __fdividef(input.z, 255.f);

	surf2Dread(&input, lutInpuSurf, color.z * sizeof(uchar4), 0);
	float b_ = __fdividef(input.w , 255.f);


	float y = __fmul_rn(0.299, r_) + __fmul_rn(0.587, g_) + __fmul_rn(0.114 , b_);
	float u = __fmul_rn(0.492 , (b_ - y));
	float v = __fmul_rn(0.877 , (r_ - y));

	surf2Dread(&input, lutInpuSurf, (int)(y * 255.f)  * sizeof(uchar4), 0);
	float lut = __fdividef(input.x, 255.f);

	y = lut;

	float r = __fmaf_rn(1.140, v, y );
	float b = __fmaf_rn(2.033, u, y );
	float g = __fmul_rn(1.704 , (y - __fmul_rn(0.299 , r) - __fmul_rn(0.114 , b)));

	return make_uchar3( fminf(__fmul_rn(r , 255.f), 255.f), fminf(__fmul_rn(g , 255.f), 255.f), fminf(__fmul_rn(b , 255.f), 255.f));
}


__device__ __inline__ bool isActive(cudaTextureObject_t boudaryTex, float2 uv, int view, int viewCnt)
{
	/*uchar1 src0;
	surf2Dread(&src0, boudarySurf, uv.x * sizeof(uchar1), uv.y);*/

	float1 src0 = tex2D<float1>(boudaryTex, uv.x, uv.y);

	float residual = src0.x - __fdividef(view + 1, viewCnt);
	if (abs(residual) < 0.05f)
		return true;
	return false;
}


__device__ __inline__ float4 getCameraColor(float2 uv, cudaTextureObject_t colorTex, cudaTextureObject_t weightTex)
{
	float4 color = tex2D<float4>(colorTex, uv.x, uv.y);
	float1 weight = tex2D<float1>(weightTex, uv.x, uv.y);

	return make_float4(color.x, color.y, color.z, weight.x);
}

#endif //_CUDACOMMON_CUH_