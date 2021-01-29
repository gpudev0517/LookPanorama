#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;

void main(void)
{
    vec3 color = texture2D(texture, texc.xy).rgb;
	oColor = vec4(color.b, color.g, color.r, 1.0f);
}