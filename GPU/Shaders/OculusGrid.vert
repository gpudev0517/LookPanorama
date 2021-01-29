#version 430
layout(location = 0) in vec4 vertex;

uniform mat4 matrix;

void main(void)
{
    gl_Position = matrix * vertex;
}