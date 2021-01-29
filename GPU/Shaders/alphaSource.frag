#version 430
uniform sampler2D texture;
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

void main(void)
{
	oColor = texture2D(texture, texc);
}