#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;
uniform int width;
uniform int height;
uniform int bytesPerLine;

void main(void)
{
	int x = int(texc.x * width);
	int y = int(texc.y * height);
	
	int x1, y1, x2, y2;
	x1 = int(x / 2) * 2;
	x2 = x1 + 1;
	y1 = int(y / 2) * 2;
	y2 = y1 + 1;
	
	vec2 texR = vec2(x1 * 1.0f / bytesPerLine, y1 * 1.0f / height);
	vec2 texB = vec2(x2 * 1.0f / bytesPerLine, y2 * 1.0f / height);
	vec2 texG1 = vec2(x2 * 1.0f / bytesPerLine, y1 * 1.0f / height);
	vec2 texG2 = vec2(x1 * 1.0f / bytesPerLine, y2 * 1.0f / height);
	
	float r = texture2D(texture, texR).x;
	float g1 = texture2D(texture, texG1).x;
	float g2 = texture2D(texture, texG2).x;
	float b = texture2D(texture, texB).x;
	
	float g = (g1 + g2) / 2;
	
	//oColor = vec4(r, g, b, 1.0f);
	oColor = vec4(g, g, b, 1.0f);
}