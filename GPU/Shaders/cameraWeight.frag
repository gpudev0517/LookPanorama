#version 430
uniform float imageWidth;
uniform float imageHeight;

uniform float blendingFalloff;
uniform float fisheyeLensRadiusRatio1;
uniform float fisheyeLensRadiusRatio2;
uniform float xrad1;
uniform float xrad2;
uniform float yrad1;
uniform float yrad2;
uniform float blendCurveStart;

#define   LensModel_standard      0
#define   LensModel_fisheye       1
#define   LensModel_opencv_standard 3
#define   LensModel_opencv_fisheye 4
uniform   int   lens;

layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

float curve(float s)
{
	return s;
}


float getRadialDistance2(vec2 imageUV)
{
	float dist2 = 0.0f;
	
	if(lens == LensModel_fisheye || lens == LensModel_opencv_fisheye)
	{
		float xdist = (imageWidth/2-imageUV.x) / (imageWidth/2);
		if(imageUV.x < imageWidth/2)
			xdist = xdist / xrad1;
		else
			xdist = xdist / xrad2;
		
		
		float ydist = (imageHeight/2-imageUV.y) / (imageHeight/2);
		if (imageUV.y > imageHeight/2)
			ydist = ydist / yrad1;
		else
			ydist = ydist / yrad2;
		
		
		if(xdist < 0) xdist = -xdist;
		if(ydist < 0) ydist = -ydist;
		
		dist2 = 1;
		if(xdist >= ydist)
		{
			dist2 = curve(xdist);
		}
		else
		{
			dist2 = curve(ydist);
		}
	}
	else if(lens == LensModel_standard || lens == LensModel_opencv_standard )
	{
		float xdist = (imageWidth/2-imageUV.x) / (imageWidth/2 * fisheyeLensRadiusRatio1);
		float ydist = (imageHeight/2-imageUV.y) / (imageHeight/2 * fisheyeLensRadiusRatio2);
		if(xdist < 0) xdist = -xdist;
		if(ydist < 0) ydist = -ydist;
		
		dist2 = 1;
		if(xdist >= ydist)
		{
			dist2 = curve(xdist);
		}
		else
		{
			dist2 = curve(ydist);
		}
	}
	dist2 = pow(dist2, 1.0f / 2 * blendingFalloff);
	
	// convert [0, start, end=1.0f] to [0, 0, 1]
	float start = blendCurveStart;
	float end = 1.0f;
	if (start >= end) start = 0.0f;
	dist2 = clamp((dist2-start) * (1.0f/(end-start)), 0.0f, 1.0f);
   
   return dist2;
}


void main()
{
	vec2 camera = vec2(texc.x * imageWidth, texc.y * imageHeight);
	float radialDist2 = 1.0f - getRadialDistance2(camera);
	
	oColor = vec4(radialDist2);
}