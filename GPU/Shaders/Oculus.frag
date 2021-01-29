#version 430
layout(location = 0) in vec2 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;
uniform int bForRift;

void main(void)
{
   	vec4 iColor = texture2D(texture, texc);
   	if(bForRift > 0)
   		oColor = vec4(pow(iColor.rgb, vec3(2.0)), iColor.a); // RGB -> sRGB
   	else
   		oColor = vec4(pow(iColor.rgb, vec3(0.5)), iColor.a); //sRGB->RGB
}