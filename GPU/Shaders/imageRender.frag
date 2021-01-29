#version 430
layout(location = 0) in vec2 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D texture;

void main(void)
{
   vec2 uv = vec2(texc.x, 1.0f - texc.y);
   oColor = texture2D(texture, uv);
}