#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D textures[2];
uniform bool isStereo;
uniform bool isOutput;

#define   LeftView 		 0
#define   RightView      1

void main(void)
{
	vec2 uv = vec2(texc.x, texc.y);
	
	bool isRight = false;
	if(isStereo)
	{
		if(uv.y < 0.5f)
		{
			uv.y *= 2;
		}
		else
		{
			uv.y = (uv.y - 0.5f) * 2;
			isRight = true;
		}
	}
	
	if (isOutput)
		uv.y = 1.0f - uv.y;
	
	vec4 src;
	vec2 targetUV = uv;

	if(isRight)
	{
		src = texture2D(textures[RightView], targetUV);
	}
	else
	{
		src = texture2D(textures[LeftView], targetUV);
	}
	
	oColor = src;
}