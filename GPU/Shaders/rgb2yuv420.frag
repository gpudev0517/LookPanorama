#version 430
layout(location = 0) in vec4 texc;

layout (location = 0) out float YOut;
layout (location = 1) out float UOut;
layout (location = 2) out float VOut;

//layout (location = 0) out vec3 color;

uniform sampler2D texture;

void main(void)
{
	//float YOut, UOut, VOut;
	
	vec2 uv = vec2(texc.x, texc.y);
	vec4 src = texture2D(texture, uv);
	int r = int(src.x * 255);
	int g = int(src.y * 255);
	int b = int(src.z * 255);
	
	int y = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
	int u = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
	int v = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
	
	YOut = y / 255.0f;
	UOut = u / 255.0f;
	VOut = v / 255.0f;
	
	//gl_FragColor = vec4(YOut, UOut, VOut, 1.0f);
	//color = vec3(YOut, UOut, VOut);
}