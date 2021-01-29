#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;

void main(void)
{
	int yuvIndex = 0;
	vec2 pixTex;
	
	float x = texc.x;
	float y = texc.y;
	if (y <= 2.0f / 3)
	{
		yuvIndex = 0;
		pixTex.y = y / 2.0f * 3;
		pixTex.x = x;
	}
	else
	{
		pixTex.y = (y - 2.0f / 3) * 3;
		if (x <= 0.5f)
		{
			yuvIndex = 1;
			pixTex.x = x * 2;
		}
		else
		{
			yuvIndex = 2;
			pixTex.x = (x - 0.5f) * 2;
		}
	}
	
	vec3 color = texture2D(texture, pixTex).rgb;
	
	int b = int(color.x * 255);
	int g = int(color.y * 255);
	int r = int(color.z * 255);
	
	int value;
	
	if (yuvIndex == 0)
		value = int((0.257 * r) + (0.504 * g) + (0.098 * b)) + 16;
	else if(yuvIndex == 1)
		value = int((0.439 * r) - (0.368 * g) - (0.071 * b)) + 128;
	else
		value = int(-(0.148 * r) - (0.291 * g) + (0.439 * b)) + 128;
	
	oColor = vec4(value / 255.0f);
}