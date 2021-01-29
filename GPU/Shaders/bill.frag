#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D bgTexture;
uniform sampler2D bannerTexture;

uniform int widthPixels;
uniform int heightPixels;

uniform mat4 bannerPaiPlane; // columns 0(ux), 1(uy), 2(uz), 3(org)
uniform float bannerPaiZdotOrg; // uz dot org
uniform mat3 bannerHomography; // H matrix for paiPlane->banner texture

#define M_PI        3.14159265358979323846
#define M_PI_2      1.57079632679489661923
#define M_PI_4      0.785398163397448309616

void sphericalToCartesian(in float theta, in float phi, out vec3 cartesian) {
    cartesian.y = sin(phi);
    cartesian.x = cos(phi) * sin(theta);
    cartesian.z = cos(phi) * cos(theta);
}

void cartesianTospherical(in vec3 cartesian, out float theta, out float phi) {
    float result = 1.0f;
	phi = asin(cartesian.y);
	float cosphi = pow(1.0f - (cartesian.y*cartesian.y), 0.5f);
	if (cosphi == 0.0f)
	{
		theta = 0.0f;
	}
	else
	{
		theta = atan(cartesian.x, cartesian.z);
	}
}

void XYnToThetaPhi(in float x_n, in float y_n, out float theta,  out float phi)
{
    theta = (2.0f * x_n - 1.0f) * M_PI;
    phi   = M_PI * y_n - M_PI_2;
}

void ThetaPhiToXYn(in float theta,  in float phi, out float x_n, out float y_n)
{
	x_n = (theta / M_PI + 1) / 2;
	y_n = (phi + M_PI_2) / M_PI;
}

bool UVToBannerCoord(in vec2 uv, out vec2 xy)
{
	float theta, phi;
	XYnToThetaPhi(uv.x, uv.y, theta, phi);
	
	vec3 v;
	sphericalToCartesian(theta, phi, v);
	
	float t = bannerPaiZdotOrg / dot(v, bannerPaiPlane[2].xyz);
	if (t <= 0.0f)
		return false;
	else
	{
		vec3 vp = t * v;
		vp -= bannerPaiPlane[3].xyz;
		float xp = dot(vp, bannerPaiPlane[0].xyz);
		float yp = dot(vp, bannerPaiPlane[1].xyz);
		vec3 vt = bannerHomography * vec3(xp, yp, 1.0);
		vt.xyz = vt.xyz / vt.z;
		xy = vt.xy;
		if (0.0 <= vt.x && vt.x <= 1.0 && 0.0 <= vt.y && vt.y <= 1.0)
			return true;
		else
			return false;
	}
}

void main(void)
{
	vec2 vt;
	int cornerCount = 9;
	vec2 offset[9];
	float pxWidth = 0.5f;
	offset[0] = vec2(0,0);
	offset[1] = vec2(pxWidth/widthPixels, 0);
	offset[2] = vec2(0, pxWidth/heightPixels);
	offset[3] = vec2(pxWidth/widthPixels, pxWidth/heightPixels);
	offset[4] = vec2(-pxWidth/widthPixels, 0);
	offset[5] = vec2(0, -pxWidth/heightPixels);
	offset[6] = vec2(-pxWidth/widthPixels, -pxWidth/heightPixels);
	offset[7] = vec2(pxWidth/widthPixels, -pxWidth/heightPixels);
	offset[8] = vec2(-pxWidth/widthPixels, pxWidth/heightPixels);
	bool isBanner[9];
	vec2 vts[9];
	bool isInside = true;
	bool bannerFlag;
	for (int i = 0; i < cornerCount; i++)
	{
		vec2 vt;
		isBanner[i] = UVToBannerCoord(texc.xy + offset[i], vt);
		vts[i] = vt;
		if (i == 0)
		{
			bannerFlag = isBanner[i];
		}
		else
		{
			if(bannerFlag != isBanner[i])
			{
				isInside = false;
			}
		}
	}
	if (isInside)
	{
		vec4 tBackground = texture2D(bgTexture, texc.xy);
		if(isBanner[0] == true)
		{
			vec2 vt = vts[0];
			vec4 tBanner = texture2D(bannerTexture, vt);
			oColor = mix(tBackground, tBanner, tBanner.a);
		}
		else
		{
			oColor = tBackground;
		}
		//oColor = vec4(1.0f, 1.0f, 1.0f, 1.0f);
	}
	else
	{
		vec4 color = vec4(0,0,0,0);
		for( int i = 0; i < cornerCount; i++ )
		{
			vec4 singleColor;
			if(isBanner[i] == true)
			{
				singleColor = texture2D(bannerTexture, vts[i]);
			}
			else
			{
				singleColor = texture2D(bgTexture, texc.xy + offset[i]);
			}
			color = mix(color, singleColor, singleColor.a / 9.0f);
		}
		oColor = color;
		//oColor = vec4(0.0f, 0.0f, 0.0f, 1.0f);
	}
}