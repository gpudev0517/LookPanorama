#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

#extension GL_ARB_shading_language_include : enable

uniform sampler2D texture;
uniform sampler2D editedWeightTexture;

#include "unwarp.glsl"

layout(std140) uniform cameraBuffer
{
	CameraData camData;
};

void main()
{
	vec2 camera;
    float alpha = XYnToLocal(texc.xy, camData, camera);

	oColor = vec4(0.0);
	if(alpha != 0.0)
	{
		float finalWeight = 0.0;
		vec2 uv = vec2(camera.x/camData.dimension.x, camera.y/camData.dimension.y);
		vec4 originWeight = texture2D(texture, uv);
		vec4 deltaWeight = texture2D(editedWeightTexture, uv);

		finalWeight = originWeight.r + deltaWeight.r - 0.5f;

		if(finalWeight < 0.0)
			finalWeight = 0.0;
		if(finalWeight > 1.0)
			finalWeight = 1.0;

		oColor = vec4(finalWeight);
	}
}