

#include "QTHandler.h"
#include <QImage>

QTHandler::QTHandler()
{
	m_buffer = NULL;
	m_size = 0;
}

QTHandler::~QTHandler()
{
	if (m_buffer)	free(m_buffer);
}

bool QTHandler::open(std::string fileName, int mode, int &xres, int &yres, float& frameRate, int compression, int qual)
{
	QString filename(fileName.c_str());

	m_frame = QImage(filename).convertToFormat(QImage::Format_RGB888);

	xres = m_frame.width();
	yres = m_frame.height();
	frameRate = 30;
	m_filename = fileName.c_str();
	m_size = m_frame.byteCount();
	m_buffer = (unsigned char*)malloc(m_size);
	memset(m_buffer, 0x00, m_size);

	return true;
}

bool QTHandler::close()
{
	return true;
}



bool QTHandler::read(int frameNum, AlignedImage& matFrame, QString fileName)
{
	int curSize = 0;
	
	m_frame = QImage(fileName).convertToFormat(QImage::Format_RGB888);
	curSize = m_frame.byteCount();

	if (curSize > m_size)
		memcpy(m_buffer, m_frame.constBits(), m_size);
	else
		memcpy(m_buffer, m_frame.constBits(), curSize);

	matFrame.setImage(m_frame, m_buffer);
	
	return true;
}


bool QTHandler::write(int frameNum, QImage& matFrame, int deviceNum)
{

	return true;
}