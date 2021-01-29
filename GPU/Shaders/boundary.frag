#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D textures[8];
uniform int viewCnt;

void main(void)
{
	vec2 uv = vec2(texc.x, texc.y);
	oColor = vec4(0.0);
	for( int currentIdx = 0; currentIdx < viewCnt; currentIdx++ )
	{
		vec4 src0 = texture2D(textures[currentIdx], uv);
	
		if (src0.a != 0.0)
		{
			bool isPrime = true;
			for (int i = 0; i < viewCnt; i++)
			{
				if (i != currentIdx)
				{
					vec4 src1 = texture2D(textures[i], uv);
					if(src1.a != 0.0)
					{
						if (src1.r > src0.r)
						{
							isPrime = false;
							break;
						}
					}
				}
			}
			if(isPrime)
			{
				oColor = vec4(1.0 * (currentIdx + 1) / viewCnt);
				return;
			}
		}
	}
}