#pragma once
#include <qvector.h>
#include <3DMath.h>
#include <qgenericmatrix.h>
#include <qrect.h>
#include <qsggeometry.h>
#include <qopenglfunctions.h>
#include <qstring.h>
#include <vector>
#include "3DMath.h"
#include "Structures.h"

#ifdef USE_CUDA
// CUDA includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#endif

typedef unsigned long long GPUResourceHandle;

#define  PO_PI			(3.1415927)
#define PO_TOW_PI		(PO_PI * 2)
#define PO_HALF_PI		(PO_PI / 2)

float ev2gain(float ev);
float gain2ev(float gain);
#ifndef M_PI
#define M_PI        3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2      1.57079632679489661923
#endif
#ifndef M_PI_4
#define M_PI_4      0.785398163397448309616
#endif

#define ROUND(v) ((int)(v>=0 ? (v+0.5f):(v-0.5f)))

extern int g_panoramaWidth, g_panoramaHeight;
extern bool g_useCUDA;


void pano2window(double panoW, double panoH, double windowW, double windowH, double px, double py, double& wx, double& wy, bool isStereo, bool atRightEye);
bool window2pano_1x1(double panoW, double panoH, double windowW, double windowH, double& px, double& py, double wx, double wy, bool isStereo);
void makeVtTexForRect(QRectF bounds, double aspect, QSGGeometry::TexturedPoint2D tp[], bool keepAspect, bool refineTexCoourds);
void refineTexCoordsForOutput(GLfloat* texc, int nV);
void refineTexCoordsForOutput(vec2* texc, int nV);

mat3 getCameraViewMatrix(float yaw, float pitch, float roll);
void sphericalToCartesian(float theta, float phi, vec3& cartesian);
bool cartesianTospherical(vec3 cartesian, float& theta, float& phi);
void XYnToThetaPhi(float x_n, float y_n, float& theta, float& phi);
void ThetaPhiToXYn(float theta, float phi, float& x_n, float& y_n);
float XYnToLocal(float x_n, float y_n, float scale, CameraParameters::LensType lens, float offset_x, float offset_y,
	float k1, float k2, float k3, float FoV, float FoVY, float imageWidth, float imageHeight,
	mat3 &cP, vec2& camera);

void saveTextureGL(GPUResourceHandle fbo, int w, int h, QString strSaveName, QOpenGLFunctions* gl, bool yMirror);
#ifdef USE_CUDA
void saveTextureCUDA(GPUResourceHandle fbo, int w, int h, QString strSaveName, QOpenGLFunctions* gl, bool yMirror);
#endif //USE_CUDA
mat3 findHomography(const std::vector<vec2>& src, const std::vector<vec2>& dst);

typedef void(*SaveTextureFunc)(GPUResourceHandle fbo, int w, int h, QString strSaveName, QOpenGLFunctions* gl, bool yMirror);

extern SaveTextureFunc saveTexture;


// ENUM define
enum WeightMapPaintMode {
	WEIGHTMAP_PAINT = 1,
	WEIGHTMAP_VIEW = 2,
	WEIGHTMAP_OVERLAP = 3
};


enum WeightMapEyeMode {
	DEFAULT = 0,
	LEFTMODE,
	RIGHTMODE,
	BOTHMODE, 
	MIRROR
};

#define UNDOREDO_BUFFER_SIZE 20

enum WEIGHTMAP_UNDOREDO_STATE
{
	NONE_UNDOREDO,
	STARTING_UNDOREDO,
	STARTED_UNDOREDO,
	ENDING_UNDOREDO,
	ENDED_UNDOREDO
};



struct WeightMapUndoRedo
{
	int camIndex;
	WeightMapEyeMode eyeMode;
	GPUResourceHandle leftUndoRedoTexId;
	GPUResourceHandle rightUndoRedoTexId;

	WeightMapUndoRedo(){
		camIndex = -1;
		eyeMode = WeightMapEyeMode::DEFAULT;
		leftUndoRedoTexId = rightUndoRedoTexId = -1;
	}
};

struct CameraData
{
	float cx;
	float cy;
	float offset_x;
	float offset_y;

	int lens;
	float k1;
	float k2;
	float k3;

	float FoV;
	float FoVY;

	float imageWidth;
	float imageHeight;

	mat4 cP;
	//mat3 cP;
};


enum ViewMode
{
	LIVE_VIEW = 1,
	SPHERICAL_VIEW = 2,
	STITCH_VIEW = 3
};

enum PlayMode
{
	START_MODE = 1,
	PAUSE_MODE = 2
};

enum BlendMode
{
	FEATHER_MODE = 1,
	MULTIBAND_MODE = 2,
	WEIGHTVISUALIZER_MODE = 3
};

enum OPEN_INI_ERROR
{
	FILE_NO_EXIST = -1,
	FILE_VERSION_NO_MATCH = -2
};

enum StreamingMode
{
	RTMP = 1,
	WEBRTC = 2
};

struct MousePos
{
	int x;
	int y;
	int width;
	int height;
};

class RecentInfo
{
public:
	QString		title;
	QString		fullPath;
	int			type;

	bool operator == ( const RecentInfo& other )
	{
		if( title != other.title )
			return false;

		if( fullPath != other.fullPath )
			return false;

		if( type != other.type )
			return false;

		return true;
	}

	RecentInfo& operator = ( const RecentInfo& other )
	{
		title = other.title;
		fullPath = other.fullPath;
		type = other.type;

		return *this;
	}
};

struct SessionTakeInformation
{
	QString name;
	QString filePath;
	QString startTime;
	QString	duration;
	QString comment;
};
