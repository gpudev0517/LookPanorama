#include "GPUProgram.h"
#include <sstream>

GPUProgram::GPUProgram(QObject *parent) : QObject(parent)
{
	m_fboId = m_fboTextureId = -1;
#ifdef USE_CUDA
	m_cudaTargetSurface = -1;
	m_cudaTargetTexture = -1;
	m_cudaTargetArray = NULL;
#endif
	m_initialized = false;
	m_program = NULL;
	m_vertexAttr = m_texCoordAttr = -1;
	m_fboId = -1;
	m_gl = NULL;
	m_functions_2_0 = NULL;

}


GPUProgram::~GPUProgram()
{
	if (m_initialized)
	{
		if (m_fboTextureId != -1)
			m_gl->glDeleteTextures(1, &m_fboTextureId);
		if (m_fboId != -1)
			m_gl->glDeleteFramebuffers(1, &m_fboId);

		if (m_program)
		{
			delete m_program;
			m_program = NULL;
		}

#ifdef USE_CUDA
		if (m_cudaTargetArray)
			cudaFreeArray(m_cudaTargetArray);
		if (m_cudaTargetSurface != -1)
			cudaDestroySurfaceObject(m_cudaTargetSurface);
		if (m_cudaTargetTexture != -1)
			cudaDestroyTextureObject(m_cudaTargetTexture);
		m_cudaTargetSurface = -1;
		m_cudaTargetArray = NULL;
#endif 

		m_fboId = m_fboTextureId = -1;
		
		m_program = NULL;
		m_vertexAttr = m_texCoordAttr = -1;
		m_fboId = -1;
		m_gl = NULL;
		m_functions_2_0 = NULL;

		m_initialized = false;
	}
}

void GPUProgram::initialize()
{
	m_initialized = true;
}

void GPUProgram::setGL(QOpenGLFunctions* gl, QOpenGLFunctions_2_0* functions_2_0, QOpenGLFunctions_4_3_Compatibility* functions_4_3)
{
	m_gl = gl;
	m_functions_2_0 = functions_2_0;
	m_functions_4_3 = functions_4_3;
}

GPUResourceHandle GPUProgram::getTargetGPUResource()
{
#ifdef USE_CUDA
	if (m_cudaTargetSurface != -1)
		return m_cudaTargetTexture;
#endif
	if (m_fboTextureId != -1)
		return m_fboTextureId;
	else
		return -1;
}

GPUResourceHandle GPUProgram::getTargetBuffer()
{
#ifdef USE_CUDA
	if (m_cudaTargetSurface != -1)
		return (GPUResourceHandle)m_cudaTargetArray;;
#endif
	if (m_fboId != -1)
		return m_fboId;
	return -1;
}

std::string parseInclude(std::string const & Line, std::size_t const & Offset)
{
	std::string Result;

	std::string::size_type IncludeFirstQuote = Line.find("\"", Offset);
	std::string::size_type IncludeSecondQuote = Line.find("\"", IncludeFirstQuote + 1);

	return Line.substr(IncludeFirstQuote + 1, IncludeSecondQuote - IncludeFirstQuote - 1);
}

QString getContent(QString directory, QString filename)
{
	QFile shaderFile(directory + filename);
	shaderFile.open(QIODevice::ReadOnly);
	QString shaderCode(shaderFile.readAll());

	return shaderCode;
}

QString manualInclude(QString filename, QString directory)
{
	QFile shaderFile(directory + filename);
	shaderFile.open(QIODevice::ReadOnly);
	QString shaderCode(shaderFile.readAll());
	std::string source = shaderCode.toStdString();

	std::stringstream stream;
	stream << source;

	std::string line;
	QString text;

	// Handle command line defines
	int lineCount = 0;
	while (std::getline(stream, line))
	{
		std::size_t Offset = 0;
		lineCount++;

		// Version
		Offset = line.find("#version");
		if (Offset != std::string::npos)
		{
			std::size_t CommentOffset = line.find("//");
			if (CommentOffset != std::string::npos && CommentOffset < Offset)
				continue;

			// Reorder so that the #version line is always the first of a shader text
			text = QString(line.c_str()) + QString("\n") + text + QString("//") + QString(line.c_str()) + QString("\n");
			continue;
		}

		// Include
		Offset = line.find("#include");
		if (Offset != std::string::npos)
		{
			std::size_t CommentOffset = line.find("//");
			if (CommentOffset != std::string::npos && CommentOffset < Offset)
				continue;

			std::string Include = parseInclude(line, Offset);

			QString Source = getContent(directory, QString(Include.c_str()));

			text += Source;
			continue;
		}

		text += QString(line.c_str()) + "\n";
	}
	return text;
}

void GPUProgram::ADD_SHADER_FROM_CODE(QOpenGLShaderProgram* program, QString type, QString res)
{
	program->addShaderFromSourceCode(
		type == QString("frag") ? QOpenGLShader::Fragment :
		type == QString("geom") ? QOpenGLShader::Geometry :
		QOpenGLShader::Vertex,
		manualInclude(QString(res) + QString(".") + QString(type), QString(":/GPU/Shaders/")));
}