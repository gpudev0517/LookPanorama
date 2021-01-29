#version 430
layout(location = 0) in vec4 vertex;
layout(location = 1) in vec4 texCoord;
layout(location = 0) out vec2 texc;

uniform mat4 matrix;

void main(void)
{
    gl_Position = matrix * vertex;
    texc = texCoord.xy;
}
