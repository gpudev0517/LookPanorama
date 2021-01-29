#version 430
layout(location = 0) in vec4 texc;
layout(location = 0) out vec4 oColor;

uniform sampler2D colorTexture;

// lut
uniform sampler2D lutTexture;

// color correction parameters
uniform vec3 ctLightColor;

// placement
uniform mat3 placeMat;

// seam
uniform sampler2D seamTexture;
uniform bool seamOn;

#define M_PI        3.14159265358979323846
#define M_PI_2      1.57079632679489661923
#define M_PI_4      0.785398163397448309616

void sphericalToCartesian(in float theta, in float phi, out vec3 cartesian) {
    cartesian.y = sin(phi);
    cartesian.x = cos(phi) * sin(theta);
    cartesian.z = cos(phi) * cos(theta);
}

float cartesianTospherical(in vec3 cartesian, out float theta, out float phi) {
    float result = 1.0f;
	phi = asin(cartesian.y);
	float cosphi = pow(1.0f - (cartesian.y*cartesian.y), 0.5f);
	if (cosphi == 0.0f)
	{
		theta = 0.0f;
		return 0.0f;
	}
	else
	{
		theta = atan(cartesian.x, cartesian.z);
		return 1.0f;
	}
}

// return float instead of bool because bool / int do not work on MacBookPro 10.6
float XYnToThetaPhi(in float x_n, in float y_n, out float theta,  out float phi)
{
    float result = 1.0f;
    theta = (2.0f * x_n - 1.0f) * M_PI;
    phi   = M_PI * y_n - M_PI_2;

    return result;
}

// return float instead of bool because bool / int do not work on MacBookPro 10.6
float ThetaPhiToXYn(in float theta,  in float phi, out float x_n, out float y_n)
{
    float result = 1.0f;
	x_n = (theta / M_PI + 1) / 2;
	y_n = (phi + M_PI_2) / M_PI;
    return result;
}

float XYnToDstXYn(in vec2 xyn, out vec2 dstXyn) {
    float result = 1.0;

    float theta, phi;
    vec3 cartesian;

    result *= XYnToThetaPhi(xyn.x, xyn.y, theta, phi);
    sphericalToCartesian(theta, phi,cartesian);
	vec3 U = placeMat * cartesian;
	cartesianTospherical(U, theta, phi);
	float x, y;
	ThetaPhiToXYn(theta, phi, x, y);
	dstXyn.x = x;
	dstXyn.y = y;
    return result;
}

vec3 calcLut(vec3 color)
{

	float r_ = texture2D(lutTexture, vec2(color.r, 0.0)).g;
	float g_ = texture2D(lutTexture, vec2(color.g, 0.0)).b;
	float b_ = texture2D(lutTexture, vec2(color.b, 0.0)).a;

	float y = 0.299 * r_ + 0.587 * g_ + 0.114 * b_;
	float u = 0.492 * (b_ - y);
	float v = 0.877 * (r_ - y);

	float lut = texture2D(lutTexture, vec2(y, 0.0)).r;

	y = lut;

	float r = y + 1.140 * v;
	float b = y + 2.033 * u;
	float g =  1.704 * ( y - 0.299 * r - 0.114 * b);

	return vec3(r, g, b);
}

void main()
{
	vec2 dstUV;
	XYnToDstXYn(texc.xy, dstUV);

	bool isSeam = false;
	if (seamOn)
	{
		float seam = texture2D(seamTexture, dstUV).r;
		if (seam != 0.0)
			isSeam = true;
	}
	if (!isSeam)
	{
		vec4 cSrc = texture2D(colorTexture, dstUV);
		cSrc.xyz = calcLut(cSrc.xyz);
		cSrc.rgb = cSrc.rgb * ctLightColor;
		oColor = cSrc;
	}
	else
		oColor = vec4(1,0,0,1);
}