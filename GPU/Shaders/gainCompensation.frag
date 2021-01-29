#version 430

uniform sampler2D texture;
uniform float gain;

layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

vec3 gainCompensation(vec3 color)
{
	int r = int(color.r * 255);
	int g = int(color.g * 255);
	int b = int(color.b * 255);
	
	int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
	int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
	int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
	
	float y_=1.1643*(y/255.0-0.0625);
	float u_=u/255.0-0.5;
	float v_=v/255.0-0.5;
	y_ = y_ * gain;

	float r_=y_+1.5958*v_;
	float g_=y_-0.39173*u_-0.81290*v_;
	float b_=y_+2.017*u_;
	
	return vec3(r_,g_,b_);
}

void main(void)
{
	vec4 src = texture2D(texture, texc.xy);
	src.xyz = gainCompensation(src.xyz);
	oColor = src;
}