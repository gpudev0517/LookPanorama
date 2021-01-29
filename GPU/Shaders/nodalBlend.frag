#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D foregroundTexture;
uniform sampler2D bgTexture[8];
uniform sampler2D bgWeightTexture[8];
uniform bool bgWeightOn[8];
uniform int bgCnt;
uniform int nodalWeightOn;

vec4 getCameraColor(int i, vec2 uv)
{
	if (bgWeightOn[i])
		return vec4(texture2D(bgTexture[i], uv).xyz, texture2D(bgWeightTexture[i], uv).r);
	else
		return vec4(texture2D(bgTexture[i], uv).xyz, 1.0);
}

void main(void)
{
	vec2 uv = vec2(texc.x, texc.y);

	// Get foreground
	vec3 fgColor = texture2D(foregroundTexture, uv).rgb;
	float fgWeight = texture2D(foregroundTexture, uv).a;

	// Get background
	vec3 bgColorSum = vec3(0.0f, 0.0f, 0.0f);
	float bgWeightSum = 0.0f;
	
	for (int i=0; i<bgCnt;  i++) {
		vec4 bgSrc = getCameraColor(i, uv);
			bgColorSum = bgColorSum + bgSrc.rgb * bgSrc.a;
			bgWeightSum = bgWeightSum + bgSrc.a;
	}
#if 0
	bgColorSum = bgColorSum / bgWeightSum;
#endif

	// Composite
	vec4 result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	vec3 csum = bgColorSum + fgColor * fgWeight;
#if 0
	vec3 csum = bgColorSum * bgWeightSum + fgColor * fgWeight;
#endif
	float ws = bgWeightSum + fgWeight;

	if(ws == 0.0f)
		result = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	else
		result = vec4(csum / ws, ws);
	
	oColor = result;
}