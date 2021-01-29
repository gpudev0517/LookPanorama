#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture1;
uniform sampler2D texture2;

void main(void)
{
	vec4 val1 = texture2D(texture1, texc.xy);
	vec4 val2 = texture2D(texture2, texc.xy);
	oColor =  vec4(val1.xyz + val2.xyz - vec3(0.5f,0.5f,0.5f), 1.0f);
}