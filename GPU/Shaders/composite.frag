#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D colorTextures[8];
uniform sampler2D weightTextures[8];
uniform int viewCnt;

//#include "unwarp.glsl"

vec4 getCameraColor(int i, vec2 uv)
{
	/*vec2 camera;
    float alpha = XYnToLocal(uv, 1.0f, camera);
	if(alpha != 0.0)
	{
		vec2 cuv(camera.x/imageWidth, camera.y/imageHeight);
		return vec4(texture2D(colorTextures[i], cuv).xyz, texture2D(weightTextures[i], cuv).r);
	}
	else
	{
		return vec4(0,0,0,0);
	}*/
	return vec4(texture2D(colorTextures[i], uv).xyz, texture2D(weightTextures[i], uv).r);
}

void main(void)
{
	vec2 uv = vec2(texc.x, texc.y);
	float ws = 0.0f;
	vec3 csum = vec3(0.0f, 0.0f, 0.0f);
	for( int i = 0; i < viewCnt; i++ )
	{
		vec4 src = getCameraColor(i, uv);
		csum = csum + src.rgb * src.a;
		ws = ws + src.a;
	}
	
    if(ws == 0.0f)
       csum = vec3(0.0f, 0.0f, 0.0f);
    else
       csum = csum / ws;
    
    oColor = vec4(csum, ws);
}