#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D boundaryTexture;
uniform int viewCnt, viewIdx;

bool isActive(vec2 uv, int view)
{
	vec4 src0 = texture2D(boundaryTexture, uv);

	float residual = src0.r - 1.0 * (view + 1) / viewCnt;
	if (abs(residual) < 0.05)
		return true;
	return false;
}

void main(void)
{
	vec2 uv = vec2(texc.x, texc.y);
	float r = 0.0;
	oColor = vec4(0.0);

	if (viewIdx == 0)
	{
		oColor = texture2D(boundaryTexture, uv);
		/*for(int i = 0; i < viewCnt; i++)
		{
			if (isActive(uv, i))
				oColor = vec4(1.0 * (i + 1) / viewCnt, 0.0, 0.0, 1.0);
		}*/
	}
	else
	{
		if (isActive(uv, viewIdx-1))
			oColor = vec4(1.0);
	}
}