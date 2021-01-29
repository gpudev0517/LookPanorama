#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D original;
uniform sampler2D gaussian;

void main(void)
{
	vec4 org = texture2D(original, texc.xy);
	vec4 gauss = texture2D(gaussian, texc.xy);
	if (org.w == 0.0f)
		oColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
	else
		oColor = vec4(org.xyz - gauss.xyz + vec3(0.5f, 0.5f, 0.5f), 1.0f);
}