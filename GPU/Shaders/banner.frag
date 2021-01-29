#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D bgTexture;
uniform sampler2D bannerTexture;

void main(void)
{
	vec4 bg = texture2D(bgTexture, texc.xy);
	vec4 banner = texture2D(bannerTexture, texc.xy);

	if (banner.a==0.0)
	{
		oColor = bg;
	}
	else
	{
		oColor = vec4(banner.rgb*banner.a + bg.rgb*(1.0-banner.a), bg.a);
	}
}