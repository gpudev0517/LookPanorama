#define M_PI        3.14159265358979323846
#define M_PI_2      1.57079632679489661923
#define M_PI_4      0.785398163397448309616

/////////////////////////////////
// camera parameters
#define   LensModel_standard      0
#define   LensModel_fisheye       1
#define   LensModel_opencv_standard 3
#define   LensModel_opencv_fisheye 4

// 16 byte padding
struct CameraData
{
	vec2 principal;
	vec2 offset;
	
	int lens;
	float k1;
	float k2;
	float k3;
	
	vec2 fov;
	vec2 dimension;

	mat4 cP;
};

void sphericalToCartesian(in float theta, in float phi, out vec3 cartesian) {
    cartesian.y = sin(phi);
    cartesian.x = cos(phi) * sin(theta);
    cartesian.z = cos(phi) * cos(theta);
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
float toLocal(in vec3 cartesian, in CameraData cam, out vec2 camera)
{
    vec3 U = (cam.cP * vec4(cartesian, 1)).xyz;
	
	float alpha = 1.0;
	
	bool opencvLens = cam.lens >= LensModel_opencv_standard;
	if (opencvLens)
	{
		if (cam.lens == LensModel_opencv_standard)
		{
			float focal = cam.dimension.x / 2 / tan(cam.fov.x / 2 * M_PI / 180.0f);
			float a = U[0] / U[2];
			float b = U[1] / U[2];
			if (U[2] <= 0.001f)
			{
				alpha = 0.0f;
			}
			else
			{
				float r2 = a*a + b*b;
				float dist1 = (1 + (cam.k1 + (cam.k2 + cam.k3*r2)*r2)*r2);
				
				if (dist1 >= 0.7)
					camera = vec2(a, b) * dist1 * focal;
				else
					alpha = 0.0;
			}
		}
		else if (cam.lens == LensModel_opencv_fisheye)
		{
			float focalX = cam.dimension.x / 2 / tan(cam.fov.x / 2 * M_PI / 180.0f);
			float focalY = cam.dimension.x / 2 / tan(cam.fov.y / 2 * M_PI / 180.0f);
			float a = U[0] / U[2];
			float b = U[1] / U[2];
			if (U[2] <= 0.001f)
			{
				alpha = 0.0f;
			}
			else
			{
				float r = sqrt(a*a + b*b);
				float theta = atan(r);
				float theta2 = theta*theta;
				float theta_distortion = theta * (1 + (cam.k1 + (cam.k2 + cam.k3*theta2)*theta2)*theta2);
				
				float x_ = theta_distortion / r * a;
				float y_ = theta_distortion / r * b;
				
				camera = vec2(focalX * x_, focalY * y_);
			}
		}
		camera = camera + cam.principal + cam.offset;
	}
	else
	{
		float theta = acos( U[2] / sqrt(U[0]*U[0] + U[1]*U[1] + U[2]*U[2]) );
		
		float x_c = 0;
		float y_c = 0;
		float r = 0;
		float fisheye_radius = cam.dimension.y;
		float fovHalfInRad = cam.fov.x/180*M_PI / 2;
		if(cam.lens==LensModel_fisheye)
		{
			// for equidistant
			float f = cam.dimension.x / 2 / fovHalfInRad;
			r = f * theta;
			
			// fisheye equisolid
			//float f = dimension.x / 4 / sin(FoV/180*M_PI/4);
			//float r = 2 * f * sin(theta/2);
			
			// fisheye stereographic
			//float f = dimension.x / 4 / tan(FoV/180*M_PI/4);
			//float r = 2 * f * tan(theta/2);
			
			// orthogonal
			//float f = dimension.x/2 / sin(FoV/180*M_PI / 2);
			//float r = f * sin(theta);
		}
		else if(cam.lens == LensModel_standard)
		{
			// Standard
			float f = cam.dimension.x/2 / tan(fovHalfInRad);
			r = f * tan(theta);
		}	
		
		float r0 = min(cam.dimension.x, cam.dimension.y) / 2.0f;
		float asp = max(cam.dimension.x, cam.dimension.y) / min(cam.dimension.x, cam.dimension.y);
		float rt = r / r0;
		if (cam.lens == LensModel_fisheye)
		{
			rt = clamp(rt, 0, asp);
		}
		else if (cam.lens == LensModel_standard)
		{
			if (rt < 0)
			{
				alpha = 0.0f;
				rt = 0;
			}
		}
		float distScale = ((cam.k1*rt+cam.k2)*rt+cam.k3)*rt + (1-cam.k1-cam.k2-cam.k3);
		if( distScale < 0.1f)
			distScale = 0.1f;
		float rc = r*distScale;
		
		float r2 = sqrt(U[0]*U[0] + U[1]*U[1]);
		float dx = U[0];
		float dy = U[1];
		if(r2 != 0)
		{
			dx /= r2;
			dy /= r2;
		}
		
		camera = vec2(dx, dy) * rc + vec2(cam.offset.x, -cam.offset.y) + cam.principal;
		if(rt == 0.0)
			alpha = 0.0;
	}
	
	if( (camera.x<0.0) || (camera.x>=cam.dimension.x) || (camera.y<0.0) || (camera.y>=cam.dimension.y) )
		alpha = 0.0;
	

    return alpha;
}

float XYnToLocal(in vec2 xyn, in CameraData cam, out vec2 camera) {
	float theta, phi;
    vec3 cartesian;

    XYnToThetaPhi(xyn.x, xyn.y, theta, phi);
    sphericalToCartesian(theta, phi,cartesian);

    // cartesian to camera
    return toLocal(cartesian, cam, camera);
}