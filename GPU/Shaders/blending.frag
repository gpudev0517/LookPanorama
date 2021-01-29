#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D colorMap[8];
uniform sampler2D blendMap[8];
uniform float diffLevelScale;
uniform float levelScale;
uniform int viewCnt;
uniform int alphaType; // 0 if blendMap is 0~1 weight map, and 1 if mask map


float getMask(vec2 uv, int view)
{
	vec4 src0 = texture2D(blendMap[0], uv);

	float residual = src0.r - 1.0 * (view + 1) / viewCnt;
	if (abs(residual) < 0.05)
		return 1.0;
	return 0.0;
}

void main(void)
{
	vec2 uv = texc.xy / levelScale;
	float ws = 0.0f;
	vec3 csum = vec3(0.0f, 0.0f, 0.0f);
	for( int i = 0; i < viewCnt; i++ )
	{
		vec3 src = texture2D(colorMap[i], uv).xyz;
		float weight;
		if (alphaType == 0)
			weight = texture2D(blendMap[i], uv * diffLevelScale).x;
		else
			weight = getMask(uv, i);
		vec3 csrc = src * weight;
		csum = csum + csrc;
		ws = ws + weight;
	}
    if(ws == 0.0f)
       csum = vec3(0.0f, 0.0f, 0.0f);
    else
       csum = csum / ws;
    
    oColor = vec4(csum, 1.0f);
}