#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;
uniform bool isVertical;
uniform int blurRadius;
uniform int width;
uniform int height;
uniform bool isPartial;
uniform float alphaType;

vec4 getTextureValue(vec2 uv)
{
	if(alphaType == 0)
		return texture2D(texture, uv);
	else
	{
		float residual = texture2D(texture, uv).r - alphaType;
		if (abs(residual) < 0.05)
		{
			return vec4(1, 0, 0, 1);
		}
		else
		{
			return vec4(0, 0, 0, 0);
		}
	}
}

void main(void)
{
	if (isPartial)
	{
		vec4 src = getTextureValue(texc.xy);
		if (src.w == 0.0f)
		{
			oColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
			return;
		}
	}
	
	vec4 dst4 = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	int pixelCnt = 0;
	if(isVertical)
	{
		for (int i = -blurRadius; i <= blurRadius; i++)
		{
			vec2 vTexUV = vec2(texc.x, texc.y + 1.0f * i / height);
			if (vTexUV.y < 0) continue;
			if (vTexUV.y > 1) continue;
			vec4 pix = getTextureValue(vTexUV);
			if (!isPartial)
			{
				dst4 += pix;
				pixelCnt++;
			}
			else if (pix.w != 0.0f)
			{
				dst4 += vec4(pix.xyz, 1.0f);
				pixelCnt++;
			}
		}
	}
	else
	{
		for (int i = -blurRadius; i <= blurRadius; i++)
		{
			vec2 vTexUV = vec2(texc.x + 1.0f * i / width, texc.y);
			if (vTexUV.x < 0) continue;
			if (vTexUV.x > 1) continue;
			vec4 pix = getTextureValue(vTexUV);
			if (!isPartial)
			{
				dst4 += pix;
				pixelCnt++;
			}
			else if (pix.w != 0.0f)
			{
				dst4 += vec4(pix.xyz, 1.0f);
				pixelCnt++;
			}
		}
	}	
	if (pixelCnt == 0)
	{
		oColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	}
	else
	{
		oColor = dst4 / pixelCnt;
	}
}