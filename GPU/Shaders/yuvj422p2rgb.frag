#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;

uniform int bytesPerLine;
uniform int width;
uniform int height;

void main(void)
{	
	float x = texc.x * width;
	float y = texc.y * height;
	vec2 texY = vec2(x * 1.0f / bytesPerLine, texc.y  / 2);
	vec2 texU = vec2(x * 1.0f / 2 / bytesPerLine, (2.0f + y * 1.0f / height) / 4);
	vec2 texV = vec2(x * 1.0f / 2 / bytesPerLine, (3.0f + y * 1.0f / height) / 4);
	
	if (mod(y, 2) == 1)
	{
		texU.x += 0.5f;
		texV.x += 0.5f;
	}
	
	
	float y_ = texture2D(texture, texY).x;
	float u = texture2D(texture, texU).x;
	float v = texture2D(texture, texV).x;
	
	u=u-0.5;
	v=v-0.5;

	float r=y_ + 1.403 * v;
	float g=y_ - 0.344* u - 0.714 * v;
	float b=y_ + 1.770 * u;
    
    oColor = vec4(r, g, b, 1.0f);
}