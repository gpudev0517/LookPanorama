#pragma once

#include <QImage>

struct AlignedImage
{
	AlignedImage()
	{
		clear();
	}

	~AlignedImage()
	{
	}

	void setImageAttribute(QImage img)
	{
		width = img.width();
		height = img.height();
		stride = img.bytesPerLine();
	}

	void setImage(QImage img, unsigned char* imgBuffer = 0)
	{
		if (imgBuffer == NULL)
			buffer = img.bits();
		else
			buffer = imgBuffer;
		setImageAttribute(img);
	}

	void makeBuffer(int len)
	{
		buffer = new unsigned char[len];
		hardCopy = true;
	}

	void dispose()
	{
		if (hardCopy)
		{
			if (buffer)
			{
				delete[] buffer;
			}
		}
	}

	unsigned char * buffer;
	int width;
	int height;
	int stride;
	bool hardCopy;

	void clear()
	{
		buffer = 0;
		width = 0;
		height = 0;
		stride = 0;
		hardCopy = false;
	}
	bool isValid() const
	{
		return buffer != NULL ? true : false;
	}
};

struct ImageBufferData
{
	AlignedImage mImageY;
	AlignedImage mImageU;
	AlignedImage mImageV;
	unsigned int mFrame;
	int msToWait;

	enum FrameFormat
	{
		NONE,
		YUV420,
		YUV422,
		YUVJ422P,
		RGB888,
		RGBA8888,
		BGR0,
		GRAYSCALE,
		BAYERRG8,
		LIDAR
	} mFormat;

	ImageBufferData(FrameFormat format = NONE)
	{
		mFormat = format;
		mFrame = 0;
		msToWait = 0;
	}
	void clear()
	{
		mImageY.clear();
		mImageU.clear();
		mImageV.clear();
		mFrame = 0;
		msToWait = 0;
		mFormat = NONE;
	}
	bool isValid() const
	{
		if (mFormat != NONE && mImageY.isValid())
			return true;
		return false;
	}

	void dispose()
	{
		mImageY.dispose();
		//mImageU.dispose();
		//mImageV.dispose();
	}
};