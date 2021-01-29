#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;
uniform int width, height;

void main(void)
{
	float src0 = texture2D(texture, vec2(texc.x, texc.y)).r;
	float src1 = texture2D(texture, vec2(texc.x + 1.0f / width, texc.y)).r;
	float src2 = texture2D(texture, vec2(texc.x - 1.0f / width, texc.y)).r;
	float src3 = texture2D(texture, vec2(texc.x, texc.y - 1.0f / height)).r;
	float src4 = texture2D(texture, vec2(texc.x, texc.y + 1.0f / height)).r;

	if((src0 == src1)&&(src0 == src2)&&(src0 == src3)&&(src0 == src4)) 
		oColor = vec4(0.0, 0.0, 0.0, 0.0);
	else
		oColor = vec4(1.0, 0.0, 0.0, 1.0);
}