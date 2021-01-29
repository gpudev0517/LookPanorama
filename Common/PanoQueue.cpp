#include "PanoQueue.h"
extern "C" {
#include "libavutil/frame.h"
}
#include "define.h"

#define QUEUE_STATUS	QString("(cur:%5, in:%1, out:%2, throw:%3, loss:%4)").ARGN(m_inFrames).ARGN(m_outFrames).ARGN(m_throwFrames).ARGN(m_lossFrames).ARGN(m_queue.size())

PanoQueue::PanoQueue(QObject *parent, int size, int depth)
	: QObject(parent)
	, m_Name("PanoQueue")
	, m_size(size)
	, m_depth(depth)
	, m_inFrames(0)
	, m_outFrames(0)
	, m_lossFrames(0)
	, m_throwFrames(0)
	, m_bufferType(BufferType::AV_BUFFER)
{

}

PanoQueue::~PanoQueue()
{
	clear();
}


int PanoQueue::enqueue(QMap<int, PANO_BUFFER> buffer, int frameIndex) {

	QMutexLocker locker(&m_mutex);
	
	if (buffer.size() < m_size || m_inFrames >= frameIndex) {
		PANO_LOG(QString("[ERROR] Buffer count is not enough. (%1/%2) (frame:%3) %4").ARGN(buffer.size()).ARGN(m_size).ARGN(frameIndex).arg(QUEUE_STATUS));
		return -1;
	}

	if (m_queue.size() > m_depth) {
		throwFirst();
		PANO_LOG(QString("[WARNING] Throw first frames because queue full. (frame:%1) %2").ARGN(frameIndex).arg(QUEUE_STATUS));
	}
	
	m_queue.append(buffer);

	if (frameIndex - (m_inFrames + m_lossFrames) > 1) {
		m_lossFrames += frameIndex - m_inFrames + 1;
		PANO_LOG(QString("[WARNING] Some frame lossed before incoming. (frame:%1) %2").ARGN(frameIndex).arg(QUEUE_STATUS));
	}
	m_inFrames++;

	return 0;
}

int PanoQueue::enqueue(PANO_BUFFER buffer, int frameIndex, int camIndex) {
	
	QMutexLocker locker(&m_mutex);

	if (!m_map.empty() && !m_map.contains(frameIndex) && m_map.constBegin().key() > frameIndex) {
		PANO_LOG(QString("[WARNING] This frame already throwed. (frame:%1) %2").ARGN(frameIndex).arg(QUEUE_STATUS));
		freeBuffer(buffer.bufferPtr);
		return -1;
	}
	
	if (!(camIndex >= 0 && camIndex <= m_size-1) || buffer.bufferPtr == NULL || buffer.size < 0) {
		PANO_LOG(QString("[ERROR] Buffer is invalid. (camera:%1, frame:%2, size:%3, ptr:%4)").ARGN(camIndex).ARGN(frameIndex).ARGN(buffer.size).arg(buffer.bufferPtr == NULL ? "NULL" : "VALID"));
		return -1;
	}

	QMap<int, PANO_BUFFER> map;

	if (m_map.empty() || !m_map.contains(frameIndex)) {
		map[camIndex] = buffer;
		if (m_queue.size() > m_depth) {
			throwFirst();
			PANO_LOG(QString("[WARNING] Throw first frames because queue full. (frame:%1) %2").ARGN(frameIndex).arg(QUEUE_STATUS));
		}
		m_queue.append(map);	// Add new frame set
		m_inFrames++;
		m_map[frameIndex] = m_queue.size() - 1;
	}
	else {
		int queueIndex = m_map[frameIndex];
		map = m_queue.at(queueIndex);
		if (map.contains(camIndex)) {
			PANO_LOG(QString("[WARNING] This frame already input. (frame:%1, camera:%2) %3").ARGN(frameIndex).ARGN(camIndex).arg(QUEUE_STATUS));
		}
		else 
		{
			map[camIndex] = buffer;
			m_queue.replace(queueIndex, map);	// Add new one frame at frame set
		}
	}

	return 0;
}

int PanoQueue::enqueue(void* buffer, int size, int frameIndex, int camIndex) {
	
	PANO_BUFFER buff = { 0 };
	buff.bufferPtr = (byte*) buffer;
	buff.size = size;
	return enqueue(buff, frameIndex, camIndex);
}

QMap<int, PANO_BUFFER> PanoQueue::dequeue() {
	QMutexLocker locker(&m_mutex);
	QMap<int, PANO_BUFFER> map;
	if (m_queue.isEmpty())	return map;
	do {
		map = m_queue.first();
		if (map.size() < m_size) {
			PANO_DLOG(QString("[WARNING] Throw first frames because count is not enough. (frame:%1, count:%2) %3").ARGN(map.size()).ARGN(m_map.constBegin().key()).arg(QUEUE_STATUS));
			//throwFirst();
			map.clear();
			return map;
		}
	} while (map.size() < m_size && !m_queue.isEmpty());
	
	if (m_queue.isEmpty()) {
		map.clear();
		return map;
	}
	m_queue.removeFirst();
	m_map.remove(m_map.constBegin().key());
	refreshMap();	// must call this function after m_map remove
	m_outFrames++;
	PANO_DLOG(QString("[STATUS] %1").arg(QUEUE_STATUS));
	return map;
}


QMap<int, PANO_BUFFER>& PanoQueue::head() {
	QMutexLocker locker(&m_mutex);
	QMap<int, PANO_BUFFER> map;
	if (m_queue.isEmpty()) {
		return map;
	}

	return m_queue.first();
}


void PanoQueue::sync() {
	QMutexLocker locker(&m_mutex);
}


void PanoQueue::clear() {
	QMutexLocker locker(&m_mutex);

	while (!m_queue.isEmpty()) {
		QMap<int, PANO_BUFFER> map = m_queue.first();
		
		for (QMap<int, PANO_BUFFER>::const_iterator iter = map.constBegin(); iter != map.constEnd(); iter++) {
			int index = iter.key();
			PANO_BUFFER buffer = iter.value();
			freeBuffer(buffer.bufferPtr);
			buffer.size = 0;
		}

		map.clear();
		m_queue.removeFirst();
	}

	m_queue.clear();
	m_map.clear();
	m_inFrames = 0;
	m_outFrames = 0;
	m_lossFrames = 0;
}

void PanoQueue::throwFirst() {
	QMap<int, PANO_BUFFER> map = m_queue.first();
	for (QMap<int, PANO_BUFFER>::const_iterator iter = map.constBegin(); iter != map.constEnd(); iter++) {
		int index = iter.key();
		PANO_BUFFER buffer = iter.value();
		freeBuffer(buffer.bufferPtr);
		buffer.size = 0;
	}
	map.clear();
	m_queue.removeFirst();
	m_map.remove(m_map.constBegin().key());
	refreshMap();	// must call this function after m_map remove
	m_throwFrames++;
}

void PanoQueue::refreshMap() {
	if (m_map.isEmpty())	return;
	for (QMap<int, int>::const_iterator iter = m_map.constBegin(); iter != m_map.constEnd(); iter++) {
		int index = iter.key();
		m_map[index]--;
	}
}

void PanoQueue::freeBuffer(void* ptr) {
	switch (m_bufferType)
	{
	case BufferType::AV_BUFFER: 
		FREE_AV_MEM((AVFrame**)(&ptr));
		FREE_MEM(ptr);
		break;
	case BufferType::CLASS_BUFFER:
		FREE_PTR(ptr);
		break;
	default:
		FREE_MEM(ptr);
		break;
	}
}