#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;

void main(void)
{
	oColor = texture2D(texture, texc.xy);
}