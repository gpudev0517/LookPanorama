#version 430
layout(location = 0) in vec2 vertex;
layout(location = 0) out float texcolor;

#extension GL_ARB_shading_language_include : enable

uniform float radius;
uniform float falloff;
uniform float strength;
uniform float centerx;
uniform float centery;

uniform float panoramaWidth;
uniform float panoramaHeight;

uniform mat4 matrix;

#include "unwarp.glsl"

layout(std140) uniform cameraBuffer
{
	CameraData camData;
};

void main(void)
{
	vec2 camera;	
    float alpha = XYnToLocal(vertex, camData, camera);

    if(alpha != 0.0)
	{
		camera = camera - camData.dimension / 2.0;
		gl_Position = matrix * vec4(camera.xy, 0, 1);

		vec2 panorama = vec2(vertex.x * panoramaWidth, vertex.y* panoramaHeight);
		float realRadius = sqrt((panorama.x - centerx) * (panorama.x - centerx) + (panorama.y - panoramaHeight + centery) * (panorama.y - panoramaHeight + centery));

		if(realRadius > radius){
			texcolor = 0.0;
		}
		else{
			float finalWeight = radius - realRadius *(1.0f - falloff);
			finalWeight /= radius;
			finalWeight *= 10.f * strength;
			finalWeight /= 255.0;
			texcolor = finalWeight;
		}
	}
	else
	{	//discard;
		gl_Position = matrix * vec4(0, 0, -10, 0);
		texcolor = 0.0;
	}
}
