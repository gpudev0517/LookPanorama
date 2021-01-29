#ifndef SLOTINFO_H
#define SLOTINFO_H

#include <QObject>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavdevice/avdevice.h>
#include <libavutil/mem.h>
#include <libswscale/swscale.h>
}

#include "Capture.h"

class SlotInfo : public QObject
{
    Q_OBJECT
public:
    explicit SlotInfo(QObject *parent = 0, int type = D360::Capture::CAPTURE_VIDEO);

	void setType(int type) { m_type = (D360::Capture::CaptureDomain)type; }
	int open(QString name);
	void close();
	int getWidth() { return m_width; }
	int getHeight() { return m_height; }
	float getRate(){ return m_rate; }

private:
	QString m_Name;
	D360::Capture::CaptureDomain m_type;
	int m_width;
	int m_height;
	float m_rate;

	AVFormatContext* m_pFormatCtx;

signals:

public slots:
};

#endif // SLOTINFO_H
