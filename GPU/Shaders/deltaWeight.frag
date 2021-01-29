#version 430
layout(location = 0) in float texcolor;
layout(location = 0) out vec4 oColor;

void main()
{
	oColor = vec4(texcolor, 0, 0, 1.0);
}