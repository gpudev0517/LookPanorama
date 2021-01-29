#version 430

#extension GL_ARB_shading_language_include : enable

layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;

#define RenderMode_Color 0
#define RenderMode_Weight 1
uniform int renderMode;

#include "unwarp.glsl"

layout(std140) uniform cameraBuffer
{
	CameraData camData;
};


void main()
{
	vec2 camera;
    float alpha = XYnToLocal(texc.xy, camData, camera);

	if(alpha != 0.0)
	{
		if (renderMode == RenderMode_Color){
			vec4 cSrc = texture2D(texture, vec2(camera.x/camData.dimension.x, camera.y/camData.dimension.y));
			cSrc = clamp(cSrc, 0.0, 1.0);
			cSrc.a = alpha;

			oColor = cSrc;
		}
		else
			oColor = vec4(alpha, 0.0f, 0.0f, alpha);
	}
	else
	{	//discard;
		oColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	}
}