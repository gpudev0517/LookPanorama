

#ifndef BUFFER_H
#define BUFFER_H
#include <memory>

#include <atomic>
#include <cstddef>

// Qt
#include <QObject>
#include <QMutex>
#include <QQueue>
#include <QSemaphore>
#include <QByteArray>
#include <QSharedPointer>



#include <QDebug>

#include <QImage>

// Local
#include "D360Parser.h"


typedef std::shared_ptr< std::map< int, QImage > > StitchFramePtr;
typedef std::shared_ptr< QImage > MatBufferPtr;

class RawImage
{
public:
	RawImage()
	{
		buffer = 0;
		width = height = 0;
	}
	RawImage(unsigned char* data, int width, int height, int bytePerPixel)
	{
		this->width = width;
		this->height = height;
		this->bytePerPixel = bytePerPixel;
		buffer = new unsigned char[width*height * bytePerPixel];
		memcpy(buffer, data, width*height * bytePerPixel);
	}
	RawImage& operator = (RawImage& other)
	{
		if (buffer)
			delete[] buffer;
		width = other.width;
		height = other.height;
		bytePerPixel = other.bytePerPixel;
		buffer = new unsigned char[width*height * bytePerPixel];
		memcpy(buffer, other.buffer, width*height * bytePerPixel);
	}
	~RawImage()
	{
		if (buffer)
			delete[] buffer;
		buffer = 0;
	}

	int width, height;
	int bytePerPixel;
	unsigned char* buffer;
};

typedef std::shared_ptr< RawImage > RawImagePtr;

class MatBuffer
{
public:
	//typedef QSharedPointer< cv::Mat > MatBufferPtr;
	

	MatBuffer()
	{
	}

	MatBuffer( const MatBuffer& other )
	{
		m_matBuffer = other.m_matBuffer;
	}

	MatBuffer& operator = (MatBuffer& other )
	{
		m_matBuffer = other.m_matBuffer;
	}
	MatBufferPtr getBuffer() 
	{
		return m_matBuffer;
	}

	void setBuffer( MatBufferPtr matBuffer )
	{
		m_matBuffer = matBuffer;
	}
protected:
	MatBufferPtr m_matBuffer;
};


Q_DECLARE_METATYPE(StitchFramePtr)
Q_DECLARE_METATYPE(MatBufferPtr)





template <typename T> class TFRingBuffer 
{
protected:
	T *m_buffer;
	std::atomic< size_t > m_head;
	std::atomic< size_t > m_tail;
	const size_t m_size;

	size_t next( size_t current )
	{
		return (current + 1) % m_size;
	}

public:

	TFRingBuffer( const size_t size ) : m_size(size), m_head(0), m_tail(0) 
	{
		m_buffer = new T[ size ];
		m_head.store( 0 );
		m_tail.store( 0 );
	}

	virtual ~TFRingBuffer() 
	{
		delete [] m_buffer;
	}

	size_t size()
	{
		return m_size;
	}

	bool push( const T &object ) 
	{
		
		size_t head = m_head.load( std::memory_order_relaxed );
		size_t nextHead = next( head );
	
		m_buffer[ head ] = object;
		m_head.store( nextHead, std::memory_order_release );

		return true;
	}

	bool pop( T &object ) 
	{
		size_t tail = m_tail.load( std::memory_order_relaxed );
		
		//m_head.load( std::memory_order_acquire );
		//std::cout << "Tail: " << tail << std::endl;
		object = m_buffer[ tail ];
		//if( object == NULL )
		//	m_head.load( std::memory_order_acquire );

		m_buffer[ tail ] = NULL;
		size_t tt = next( tail );
		//std::cout << "Storing Tail " << tt << std::endl;
		
		m_tail.store( tt, std::memory_order_release );
		return true;
	}
};



template<class T> class Buffer
{
    public:
        Buffer( int size );
		~Buffer();
        void add( const T data, bool dropIfFull = false );
		void setCamSettings( const CameraInput& camSettings )
		{
			m_camSettings = camSettings;
		}

        T get();
        CameraInput getCameraSettings()
		{
			return m_camSettings;
		}

		int   size();
        int   maxSize();
        bool  clear();
        bool  isFull();
        bool  isEmpty();

    private:
        QMutex queueProtect;
        QQueue<T> queue;
        QSemaphore *freeSlots;
        QSemaphore *usedSlots;
        QSemaphore *clearBuffer_add;
        QSemaphore *clearBuffer_get;

		TFRingBuffer<T> circularqueue;

		CameraInput m_camSettings;
        int bufferSize;
};

template<class T> Buffer<T>::Buffer( int size ): circularqueue( size )
{
    // Save buffer size
    bufferSize = size;
	//
    // Create semaphores
    //
	freeSlots = new QSemaphore( bufferSize );
    usedSlots = new QSemaphore( 0 );
    clearBuffer_add = new QSemaphore( 1 );
    clearBuffer_get = new QSemaphore( 1 );
}

template<class T> Buffer<T>::~Buffer()
{
	delete freeSlots;
	delete usedSlots;
	delete clearBuffer_add;
	delete clearBuffer_get;
}

template<class T> void Buffer<T>::add( const T data, bool dropIfFull )
{
#if 0
    // Acquire semaphore
	//
    clearBuffer_add->acquire();
    //
	// If dropping is enabled, do not block if buffer is full
    //
	if( dropIfFull )
    {
        // Try and acquire semaphore to add item
       if( freeSlots->tryAcquire() )
        {
			//
            // Add item to queue
            //
			queueProtect.lock();
            queue.enqueue( data );
            queueProtect.unlock();
            //
			// Release semaphore
            //
			usedSlots->release();
        }
    }
	//
    // If buffer is full, wait on semaphore
    else
    {
		//
        // Acquire semaphore
        //
		freeSlots->acquire();
		//
        // Add item to queue
        //
		queueProtect.lock();
		//std::cout << "Enqueue " << std::endl;
        queue.enqueue( data );
        queueProtect.unlock();
        //
		// Release semaphore
        //
		usedSlots->release();
    }
	//
    // Release semaphore
    //
	clearBuffer_add->release();
#else
	
	circularqueue.push( data );
#endif
}

template<class T> T Buffer<T>::get()
{
    //
	// Local variable(s)
	//
	
    T data;
#if 0
	//
	// Acquire semaphores
    //
	clearBuffer_get->acquire();
    usedSlots->acquire();
    
	//
	// Take item from queue
    //
	queueProtect.lock();
    data = queue.dequeue();
    queueProtect.unlock();
    
	//
	// Release semaphores
    //
	//freeSlots->release();
    clearBuffer_get->release();
    
	//
	// Return item to caller
    //
#else
	//std::cout << "Popping Data " << std::endl;
	circularqueue.pop( data );
#endif
	return data;
}

template<class T> bool Buffer<T>::clear()
{
#if 0
	//
    // Check if buffer contains items
	//
    if( queue.size() > 0 )
    {
		//
        // Stop adding items to buffer (will return false if an item is currently being added to the buffer)
        //
		if( clearBuffer_add->tryAcquire() )
        {
			//
            // Stop taking items from buffer (will return false if an item is currently being taken from the buffer)
            //
			if( clearBuffer_get->tryAcquire() )
            {
				//
                // Release all remaining slots in queue
                //
				freeSlots->release( queue.size() );
				//
                // Acquire all queue slots
                //
				freeSlots->acquire( bufferSize );
				//
                // Reset usedSlots to zero
				//
                usedSlots->acquire( queue.size() );
                //
				// Clear buffer
                //
				queue.clear();
                //
				// Release all slots
                //
				freeSlots->release( bufferSize );
				//
                // Allow get method to resume
                //
				clearBuffer_get->release();
            }
            else
                return false;
            // Allow add method to resume
            clearBuffer_add->release();
            return true;
        }
        else
            return false;
    }
    else
        return false;
#endif
	return true;
}

template<class T> int Buffer<T>::size()
{
#if 1
    return queue.size();
#else
	return circularqueue.size();
#endif
}

template<class T> int Buffer<T>::maxSize()
{
    return bufferSize;
}

template<class T> bool Buffer<T>::isFull()
{
#if 1
    return queue.size() == bufferSize;
#else
	return false;
#endif
}

template<class T> bool Buffer<T>::isEmpty()
{
#if 1
    return queue.size() == 0;
#else
	return false;
#endif
}





#endif // BUFFER_H
