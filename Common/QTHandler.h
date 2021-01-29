
#pragma once

#include <stdint.h>
#include <stdio.h>

#include "ImageHandler.h"


class QTHandler : public ImageHandler
{
public:

	QTHandler();
	virtual ~QTHandler();

	bool open(std::string fileName, int mode, int& xsize, int& ysize, float& frameRate, int mCompression = H_COMPRESSION_NONE,
		int compressionQuality = 100);
	bool close();
	bool read(int frameNum, AlignedImage& matFrame, QString fileName);
	bool write(int frameNum, QImage& matFrame, int deviceNum = 0);

protected:
	FileMode mMode;

	std::string mPrefix;

	eHCompression mCompression;
	uint          mImageSizeBytes;

	QImage m_frame;
	int m_size;
	QString m_filename;
	unsigned char* m_buffer;
};
