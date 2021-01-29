
#include <math.h>
#include "common.h"
#include <qimage.h>
// Eigen headers for homography
#include <Dense>
using namespace Eigen;

int g_panoramaWidth = 0;
int g_panoramaHeight = 0;
bool g_useCUDA = false;

SaveTextureFunc saveTexture;

float ev2gain(float ev)
{
	// experimental exposure value to gain conversion
	return 2.5778*exp(-0.315*(3 - ev));
}

float gain2ev(float gain)
{
	// gain conversion to experimental exposure value
	return log(gain / 2.5778) / 0.315 + 3;
}

void pano2window(double panoW, double panoH, double windowW, double windowH, double px, double py, double& wx, double& wy, bool isStereo, bool atRightEye)
{
	if (isStereo)
		panoH *= 2;

	double curW = windowW;
	double curH = windowH;

	if (panoH / panoW > windowH / windowW)
	{
		curH = windowH;
		curW = curH * (panoW / panoH);
	}
	else
	{
		curW = windowW;
		curH = curW * panoH / panoW;
	}
	double xOrg = (windowW - curW) / 2;
	double yOrg = (windowH - curH) / 2;

	wx = (curW / panoW) * px + xOrg;
	wy = (curH / panoH) * py + yOrg;
	if (isStereo && atRightEye)
		wy += curH / 2;
}

bool window2pano_1x1(double panoW, double panoH, double windowW, double windowH, double& px, double& py, double wx, double wy, bool isStereo)
{
	if (isStereo)
		panoH *= 2;

	double curW = windowW;
	double curH = windowH;

	if (panoH / panoW > windowH / windowW)
	{
		curH = windowH;
		curW = curH * (panoW / panoH);
	}
	else
	{
		curW = windowW;
		curH = curW * panoH / panoW;
	}
	double xOrg = (windowW - curW) / 2;
	double yOrg = (windowH - curH) / 2;

	px = (wx - xOrg) / curW;
	py = (wy - yOrg) / curH;

	bool atRightEye = false;
	if (isStereo)
	{
		if (0.5 < py)
		{
			atRightEye = true;
			py -= 0.5;
		}
		py *= 2; // py = (wy - yOrg) / (curH/2);
	}
	return atRightEye;
}

void makeVtTexForRect(QRectF bounds, double aspect, QSGGeometry::TexturedPoint2D tp[], bool keepAspect, bool refineTexCoourds)
{
	double curAspect = bounds.width() / bounds.height();

	double ox = bounds.x();
	double oy = bounds.y();
	double w = bounds.width();
	double h = bounds.height();

	if (keepAspect)
	{
		if (curAspect >= aspect)
		{
			w = h * aspect;
			ox += (bounds.width() - w) / 2;
		}
		else
		{
			h = w / aspect;
			oy += (bounds.height() - h) / 2;
		}
	}

	int i = 0;
	vec2 vt[4];
	vt[i++] = vec2(ox, oy + h);
	vt[i++] = vec2(ox + w, oy + h);
	vt[i++] = vec2(ox, oy);
	vt[i++] = vec2(ox + w, oy);

	i = 0;
	vec2 tex[4];
	tex[i++] = vec2(0, 0);
	tex[i++] = vec2(1, 0);
	tex[i++] = vec2(0, 1);
	tex[i++] = vec2(1, 1);

	if (refineTexCoourds)
		refineTexCoordsForOutput(tex, 4);

	for (int i = 0; i < 4; i++)
		tp[i].set(vt[i].x, vt[i].y, tex[i].x, tex[i].y);
}

void refineTexCoordsForOutput(GLfloat* texc, int nV)
{
	for (int i = 0; i < nV; i++)
	{
		int y = i * 2 + 1;
		texc[y] = 1 - texc[y];
	}
}

void refineTexCoordsForOutput(vec2* texc, int nV)
{
	for (int i = 0; i < nV; i++)
	{
		texc[i].y = 1 - texc[i].y;
	}
}

/// coordinates system
mat3 getCameraViewMatrix(float yaw, float pitch, float roll)
{
	// every camera's yaw, pitch, roll from panorama tools
	// yaw : ->left : is not equal with global yaw.
	// pitch : ->down
	// roll : ->rightdown
	vec3 u(roll * sd_to_rad, pitch * sd_to_rad, -yaw * sd_to_rad);

	mat3 m = mat3_id;
	m.set_rot_zxy(u);

	return m;
}

void sphericalToCartesian(float theta, float phi, vec3& cartesian)
{
	cartesian.y = sin(phi);
	cartesian.x = cos(phi) * sin(theta);
	cartesian.z = cos(phi) * cos(theta);
}

bool cartesianTospherical(vec3 cartesian, float& theta, float& phi)
{
	float result = 1.0f;
	phi = asin(cartesian.y);
	float cosphi = pow(1.0f - (cartesian.y*cartesian.y), 0.5f);
	if (cosphi == 0.0f)
	{
		theta = 0.0f;
		return false;
	}
	else
	{
		theta = atan2(cartesian.x, cartesian.z);
		return true;
	}
}

void XYnToThetaPhi(float x_n, float y_n, float& theta, float& phi)
{
	theta = (2.0f * x_n - 1.0f) * M_PI;
	phi = M_PI * y_n - M_PI_2;
}

void ThetaPhiToXYn(float theta, float phi, float& x_n, float& y_n)
{
	x_n = (theta / M_PI + 1) / 2;
	y_n = (phi + M_PI_2) / M_PI;
}

void saveTextureGL(GPUResourceHandle fbo, int w, int h, QString strSaveName, QOpenGLFunctions* gl, bool yMirror)
{
	int myDataLength = w * h * 4;
	GLuint *buffer = (GLuint *)malloc(myDataLength);
	
	gl->glBindFramebuffer(GL_FRAMEBUFFER,(GLuint) fbo);
	gl->glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, buffer);

	QImage img((uchar*)buffer, w, h, QImage::Format_RGBA8888);
	if (yMirror)
		img = img.mirrored();
	img.save(strSaveName);
	gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);
	
	free(buffer);
}

#ifdef USE_CUDA
void saveTextureCUDA(GPUResourceHandle fbo, int w, int h, QString strSaveName, QOpenGLFunctions* gl, bool yMirror = false)
{
	int myDataLength = w * h * 4;
	GLuint *buffer = (GLuint *)malloc(myDataLength);

	cudaMemcpyFromArray(buffer, (cudaArray *)fbo, 0, 0, w * h * 4, cudaMemcpyDeviceToHost);

	QImage img((uchar*)buffer, w, h, QImage::Format_RGBA8888);
	if (yMirror)
		img = img.mirrored();
	img.save(strSaveName);
	gl->glBindFramebuffer(GL_FRAMEBUFFER, 0);

	free(buffer);
}
#endif

mat3 findHomography(const std::vector<vec2>& src, const std::vector<vec2>& dst)
{
	// for the banner task, we just assumes 4 point problem.
	assert(src.size() == 4 && dst.size() == 4);
	int cp = 4;

	// Multiple View Geometry in Computer Vision, 2004, by Hartley ZisserMan, Page 106
	// 4.1. The Direct Linear Transformation (DLT) algorithm

	// xi' x Hxi = 0
	// xi, xi' : 3 by 1 homogeneous column vector
	// H : 3 by 3 homography matrix

	// Equation (4.3)

	// [0			-wi'xiT		 yi'xiT] (h1 h2 h3)' = 0
	// [wi'xiT		0			-xi'xiT]

	// Aih = 0

	// h = (h1 h2 h3)'

	// xi:
	std::vector<vec3> xi; // homogeneous coordinate of src
	for (int i = 0; i < cp; i++)
	{
		xi.push_back(vec3(src[i].x, src[i].y, 1.0f));
	}
	std::vector<vec3> xin; // homogeneous coordinate of dst
	for (int i = 0; i < cp; i++)
	{
		xin.push_back(vec3(dst[i].x, dst[i].y, 1.0f));
	}

	// Ah = 0
	// A : 8 by 9 matrix
	// h : 9 by 1 column vector
	MatrixXf A(8, 9);
	for (int i = 0; i < cp; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			A(i * 2 + 0, j) = 0;
			A(i * 2 + 0, 3 + j) = -xin[i].z * xi[i][j];
			A(i * 2 + 0, 6 + j) = xin[i].y * xi[i][j];

			A(i * 2 + 1, j) = xin[i].z * xi[i][j];
			A(i * 2 + 1, 3 + j) = 0;
			A(i * 2 + 1, 6 + j) = -xin[i].x * xi[i][j];
		}
	}

	// Solving for H
	// A has rank 8, and thus has a 1-dimensional null-space which provides a solution for h.
	// Such a solution h can only be determined up to a non-zero scale factor

	// Algorithm 4.1. (iii)
	// Obtain the SVD of A. The unit single vector corresponding to the smallest singular value is the solution h.
	// Specifically, if A = UDV' with D diagonal with positive diagonal entries, arranged in descending order down the diagonal,
	// then h is the last volumn of V.
	JacobiSVD<MatrixXf> svd(A, ComputeThinU | ComputeFullV);
	VectorXf hVector = svd.matrixV().col(8);
	mat3 H(hVector.data());
	return transpose(H);
}