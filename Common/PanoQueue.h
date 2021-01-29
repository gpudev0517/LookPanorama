#ifndef PANOQUEUE_H
#define PANOQUEUE_H

#include <QObject>
#include <QList>
#include <QMap>
#include <QMutex>
#include <QMutexLocker>

#include "define.h"
#include "Structures.h"

class PanoQueue : public QObject
{
	Q_OBJECT

public:
	PanoQueue(QObject *parent, int size = 1, int depth = 5);
	virtual ~PanoQueue();
	void setBufferType(int type) { m_bufferType = type; }
	void setSize(int size) { m_size = size; }
	void setDepth(int depth) { m_depth = depth; }

	enum BufferType
	{
		AV_BUFFER = 0,
		CLASS_BUFFER = 1,
		COMMON_BUFFER = 2
	};

	int enqueue(QMap<int, PANO_BUFFER>, int frameIndex = -1);
	int enqueue(PANO_BUFFER buffer, int frameIndex, int camIndex = 0);
	int enqueue(void* buffer, int size, int frameIndex, int camIndex = 0);
	QMap<int, PANO_BUFFER> dequeue();
	QMap<int, PANO_BUFFER>& head();
	void sync();
	void clear();
	void freeBuffer(void* ptr);

private:
	void throwFirst();
	void refreshMap();

private:
	QString m_Name;
	int	m_size;
	int m_depth;
	int m_bufferType;
	int m_inFrames;
	int m_outFrames;
	int m_lossFrames;
	int m_throwFrames;
	QList<QMap<int, PANO_BUFFER>>	m_queue;
	QMap<int, int> m_map;
	QMutex m_mutex;
	
};

#endif // PANOQUEUE_H
