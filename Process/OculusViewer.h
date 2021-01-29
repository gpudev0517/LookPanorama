#ifndef OCULUSVIEWER_H
#define OCULUSVIEWER_H

// Oculus Rift
#include "TLogger.h"
#include "Extras/OVR_Math.h"
#include "OVR_CAPI.h"
#include "OVR_CAPI_GL.h"

using namespace OVR;

#include <math.h>
#include <QTimer>
#include <QThread>
#include <QQueue>
#include <QTime>
#include <QWindow>

#include "Structures.h"
#include "SharedImageBuffer.h"

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>

// Shows checkboard grid for distortion inspection
#define DEBUG_OCULUS_VIEWER 1

//---------------------------------------------------------------------------------------
struct DepthBuffer
{
	GLuint        texId;

    DepthBuffer(Sizei size, int sampleCount)
    {
        UNREFERENCED_PARAMETER(sampleCount);

        //assert(sampleCount <= 1); // The code doesn't currently handle MSAA textures.

        glGenTextures(1, &texId);
        glBindTexture(GL_TEXTURE_2D, texId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        GLenum internalFormat = GL_DEPTH_COMPONENT24;
        GLenum type = GL_UNSIGNED_INT;
        /*if (GLE_ARB_depth_buffer_float)
        {
            internalFormat = GL_DEPTH_COMPONENT32F;
            type = GL_FLOAT;
        }*/

		glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, size.w, size.h, 0, GL_DEPTH_COMPONENT, type, NULL);
	}
	virtual ~DepthBuffer()
	{
		if (texId)
		{
			glDeleteTextures(1, &texId);
			texId = 0;
		}
	}
};


//--------------------------------------------------------------------------
#ifndef OCULUS_1_8_0
struct TextureBuffer
{
	ovrHmd              hmd;
	ovrSwapTextureSet*  TextureSet;
	GLuint              texId;
	GLuint              fboId;
	Sizei               texSize;
	QOpenGLFunctions	*functions;

	TextureBuffer(ovrHmd hmd, bool rendertarget, bool displayableOnHmd, Sizei size, unsigned char * data, QOpenGLFunctions * functions) :
		hmd(hmd),
		TextureSet(nullptr),
		texId(0),
		fboId(0),
		texSize(0, 0)
	{
		texSize = size;
		this->functions = functions;

		if (displayableOnHmd)
		{
			ovrResult result = ovr_CreateSwapTextureSetGL(hmd, GL_SRGB8_ALPHA8, size.w, size.h, &TextureSet);

			if (OVR_SUCCESS(result))
			{
				for (int i = 0; i < TextureSet->TextureCount; ++i)
				{
					ovrGLTexture* tex = (ovrGLTexture*)&TextureSet->Textures[i];
					glBindTexture(GL_TEXTURE_2D, tex->OGL.TexId);

					if (rendertarget)
					{
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
					}
					else
					{
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
					}
				}
			}
		}
		else
		{
			glGenTextures(1, &texId);
			glBindTexture(GL_TEXTURE_2D, texId);

			if (rendertarget)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			}

			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, texSize.w, texSize.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		}

		functions->glGenFramebuffers(1, &fboId);
	}

	virtual ~TextureBuffer()
	{
		if (TextureSet)
		{
			ovr_DestroySwapTextureSet(hmd, TextureSet);
			TextureSet = nullptr;
		}
		if (texId)
		{
			glDeleteTextures(1, &texId);
			texId = 0;
		}
		if (fboId)
		{
			functions->glDeleteFramebuffers(1, &fboId);
			fboId = 0;
		}
	}

	Sizei GetSize() const
	{
		return texSize;
	}

	void SetAndClearRenderSurface(DepthBuffer* dbuffer)
	{
		auto tex = reinterpret_cast<ovrGLTexture*>(&TextureSet->Textures[TextureSet->CurrentIndex]);

		functions->glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex->OGL.TexId, 0);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dbuffer->texId, 0);

		functions->glViewport(0, 0, texSize.w, texSize.h);
		functions->glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		functions->glEnable(GL_FRAMEBUFFER_SRGB);
	}

	void UnsetRenderSurface()
	{
		functions->glDisable(GL_FRAMEBUFFER_SRGB);
		functions->glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
	}
};
#else
struct TextureBuffer
{
	ovrSession          Session;
	ovrTextureSwapChain  TextureChain;
	GLuint              texId;
	GLuint              fboId;
	Sizei               texSize;
	QOpenGLFunctions	*functions;

	TextureBuffer(ovrSession session, bool rendertarget, bool displayableOnHmd, Sizei size, int mipLevels, unsigned char * data, int sampleCount, QOpenGLFunctions * functions) :
        Session(session),
        TextureChain(nullptr),
		texId(0),
		fboId(0),
		texSize(0, 0)
	{
        UNREFERENCED_PARAMETER(sampleCount);

        //assert(sampleCount <= 1); // The code doesn't currently handle MSAA textures.

        texSize = size;
		this->functions = functions;

        if (displayableOnHmd)
        {
            // This texture isn't necessarily going to be a rendertarget, but it usually is.
            //assert(session); // No HMD? A little odd.
            //assert(sampleCount == 1); // ovr_CreateSwapTextureSetD3D11 doesn't support MSAA.

            ovrTextureSwapChainDesc desc = {};
            desc.Type = ovrTexture_2D;
            desc.ArraySize = 1;
            desc.Width = size.w;
            desc.Height = size.h;
            desc.MipLevels = 1;
            desc.Format = OVR_FORMAT_R8G8B8A8_UNORM_SRGB;
            desc.SampleCount = 1;
            desc.StaticImage = ovrFalse;
            ovrResult result = ovr_CreateTextureSwapChainGL(Session, &desc, &TextureChain);

            int length = 0;
            ovr_GetTextureSwapChainLength(session, TextureChain, &length);
			if (OVR_SUCCESS(result))
			{
                for (int i = 0; i < length; ++i)
				{
                    GLuint chainTexId;
                    ovr_GetTextureSwapChainBufferGL(Session, TextureChain, i, &chainTexId);
                    glBindTexture(GL_TEXTURE_2D, chainTexId);

					if (rendertarget)
					{
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
					}
					else
					{
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
						glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
					}
				}
			}
		}
		else
		{
			glGenTextures(1, &texId);
			glBindTexture(GL_TEXTURE_2D, texId);

			if (rendertarget)
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			}
			else
			{
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			}

			glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, texSize.w, texSize.h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		}

        if (mipLevels > 1)
        {
            functions->glGenerateMipmap(GL_TEXTURE_2D);
        }
		functions->glGenFramebuffers(1, &fboId);
	}

	virtual ~TextureBuffer()
	{
        if (TextureChain)
        {
            ovr_DestroyTextureSwapChain(Session, TextureChain);
            TextureChain = nullptr;
        }
		if (texId)
		{
			glDeleteTextures(1, &texId);
			texId = 0;
		}
		if (fboId)
		{
			functions->glDeleteFramebuffers(1, &fboId);
			fboId = 0;
		}
	}

	Sizei GetSize() const
	{
		return texSize;
	}

	void SetAndClearRenderSurface(DepthBuffer* dbuffer)
	{
        GLuint curTexId;
        if (TextureChain)
        {
            int curIndex;
            ovr_GetTextureSwapChainCurrentIndex(Session, TextureChain, &curIndex);
            ovr_GetTextureSwapChainBufferGL(Session, TextureChain, curIndex, &curTexId);
        }
        else
        {
            curTexId = texId;
        }

		functions->glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, curTexId, 0);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, dbuffer->texId, 0);

		glViewport(0, 0, texSize.w, texSize.h);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_FRAMEBUFFER_SRGB);
	}

	void UnsetRenderSurface()
	{
		glDisable(GL_FRAMEBUFFER_SRGB);
		functions->glBindFramebuffer(GL_FRAMEBUFFER, fboId);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
		functions->glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0);
		
	}
    void Commit()
    {
        if (TextureChain)
        {
            ovr_CommitTextureSwapChain(Session, TextureChain);
        }
    }
};
#endif


class OculusRender : public QWindow //, public QOpenGLWidget, protected QOpenGLFunctions //public QObject ////
{
	Q_OBJECT
public:
	OculusRender(QOpenGLContext* context);
	virtual ~OculusRender();

	QOpenGLContext * context() {
		return m_context;
	}

	void makeCurrent() {
		m_context->makeCurrent(this);
	}

protected:
	virtual void updateFps(float fps) {
	}

	virtual void perFrameRender() {

	}

	virtual void perEyeRender() {
	}

public:

	void	initialize(SharedImageBuffer *sharedImageBuffer, int width, int height);
	void	setPanoramaTexture(int panoramaTexture);
	bool	isCreated(){ return m_isCreated; }
	int		getTextureID();
	int		getTextureIDForQML();
	void	showChessboard(bool isShow) { isChessboard = isShow; }

public:
	void initializeGL(bool isStereo);
	void paintGL();
	void updateGL();
	void resizeGL(int width, int height);

private:
	void Render(OVR::Matrix4f view, OVR::Matrix4f proj, int isRight);
	void RenderForQMLTexture(Matrix4f viewproj);

private:
	QOpenGLContext * m_context;

#ifndef OCULUS_1_8_0
	ovrHmd session;
	ovrGLTexture  * mirrorTexture;
#else
	ovrSession session;
	ovrMirrorTexture  mirrorTexture;
	GLuint texId;
	long long frameIndex;
#endif
	ovrHmdDesc hmdDesc;

	ovrEyeRenderDesc eyeRenderDesc[2];
	ovrVector3f hmdToEyeViewOffset[2];
	ovrLayerEyeFov layer;

	TextureBuffer * renderTexture;
	DepthBuffer * depthBuffer;
	GLuint mirrorFBO;
	GLuint mirrorFBOForQML;
	GLuint mirrorTextureForQML;

	ovrBool m_isCreated;


	// OpenGL
	int m_panoramaTexture;

	GLuint m_vertexAttr;
	GLuint m_texCoordAttr;
	int m_matrixUnif;
	int m_isRiftUnif;

	QOpenGLShaderProgram *m_program;

	GLfloat *vertices;
	GLfloat *texCoords, *texCoordsRight;

	GLfloat *verticesForQML;
	GLfloat *texCoordsForQML;

#if DEBUG_OCULUS_VIEWER
	QOpenGLShaderProgram *m_gridProgram;
	
	GLuint m_gridVertexAttr;
	int m_gridMatrixUnif;

	GLfloat *verticesGrid;
	int m_gridPointCount;
#endif

	// Panorama
	int m_panoramaWidth;
	int m_panoramaHeight;

	bool isStereo;

	bool isChessboard;

	QString m_Name;
	bool m_finished;

public slots:
	void renderFrame();
	bool close();


public:
	enum StopReason
	{
		FINISH_CONFIGURATION,
		SWITCH_OFF
	};

	// Threads
	void	start();
	void	qquit();

	void	stop(OculusRender::StopReason reason);
	bool	connect();
	bool	disconnect();
	bool	isConnected();

	bool	isFinished();
	void	setFinished() { m_finished = true; };

	void startThread();

private:

	SharedImageBuffer *sharedImageBuffer;

	QThread* oculusThreadInstance;
	QMutex doStopMutex;

	bool doStop;
	StopReason stopReason;
	bool isDisconnecting;

	QWaitCondition finishWC;

protected:
	void run();

public slots:
	void process();

signals:
	void updateStatisticsInGUI(struct ThreadStatisticsData);
	void finished(int type, QString msg, int id);
	void started(int type, QString msg, int id);
};

#endif //OCULUSVIEWER_H