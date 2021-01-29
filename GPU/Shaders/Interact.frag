#version 430
layout(location = 0) in vec2 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;
uniform int panoSection; //!< 0 for left, 1 for right, 2 for mono

void main(void)
{
	vec2 tex = vec2(texc.x, 1.0f - texc.y);
	switch(panoSection)
	{
		case 0:
			tex.y /= 2;
			break;
		case 1:
			tex.y = tex.y / 2 + 0.5f;
			break;
		case 2:
			break;
	}
	oColor = texture2D(texture, tex);
}