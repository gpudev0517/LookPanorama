#ifndef RINGBUFFER_H
#define RINGBUFFER_H

#include <QSemaphore>
#include <QDebug>
template <typename T> class RingBuffer
{
    T *m_buffer;
    const size_t m_size;

    QSemaphore freeObjects;
    QSemaphore usedObjects;

    unsigned int readPos;
    unsigned int writePos;
public:
    RingBuffer(const size_t size):
        m_size(size),
        freeObjects(size),
        usedObjects(0),
        readPos(0),
        writePos(0)
    {
        m_buffer = new T[ size ];
    }
    virtual ~RingBuffer()
    {
        delete [] m_buffer;
    }
    bool push( const T &object )
    {
        if(freeObjects.tryAcquire())
        {
            m_buffer[writePos % m_size] = object;
            writePos ++;
            usedObjects.release();
            return true;
        }

//        qDebug() << "RingBuffer: No free space";
        return false;
    }
    bool pop( T &object )
    {
        if(usedObjects.tryAcquire())
        {
            object = m_buffer[readPos % m_size];
            readPos ++;
            freeObjects.release();
            return true;
        }

//        qDebug() << "RingBuffer: No data to read";
        return false;
    }
};

#endif // RINGBUFFER_H
