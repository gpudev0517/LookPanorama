
#include <iostream>
#include "SharedImageBuffer.h"
#include "D360Stitcher.h"
#include "D360Parser.h"


SharedImageBuffer::SharedImageBuffer()
{
	liveGrabIndex = -1;
	initialize();
}

SharedImageBuffer::~SharedImageBuffer()
{
	m_stitcher = NULL;
	m_playbackStitcher = NULL;
}

void SharedImageBuffer::initialize()
{
	nArrived = 0;
	doSync = false;

	videoProcessedId = -1;
	videoPlaybackId = -1;
	audioCapturedCount = 0;
	audioProcessedId = -1;
	audioPlaybackId = -1;
	m_globalStates.clear();
	rawImages.clear();
	m_streamer = NULL;
	m_globalAnimSettings = NULL;
	isFinalizing = false;

	weightEditEnabled = false;
	selectedViewIndex1 = -1;
}

void SharedImageBuffer::initializeForReplay()
{
	nArrived = 0;
	doSync = false;

	videoProcessedId = -1;
	videoPlaybackId = -1;
	audioCapturedCount = 0;
	audioProcessedId = -1;
	audioPlaybackId = -1;
	m_globalStates.clear();
	rawImages.clear();

	wakeAll();
}

void SharedImageBuffer::setSeekFrames(int nFrames)
{
	videoPlaybackId = nFrames - 1;
	audioPlaybackId = nFrames - 1;
}

void SharedImageBuffer::add(int deviceNumber, bool sync)
{
	//
	// Device stream is to be synchronized
	//
	if (sync)
	{
		mutex.lock();
		doSync = sync;
		//std::cout << "Inserting " << deviceNumber << std::endl;
		syncSet.insert(deviceNumber);
		mutex.unlock();
	}
}

void SharedImageBuffer::addState(int deviceNumber, GlobalState& state, bool sync)
{
	//
	// If File Streams should be synced
	//
	if (sync)
	{
		mutex.lock();
		syncSet.insert(deviceNumber);
		mutex.unlock();
	}
	m_globalStates[deviceNumber] = state;
}

void SharedImageBuffer::setPlaybackRawImage(ImageDataPtr image)
{
	playbackRawImage = image;
}

void SharedImageBuffer::setRawImage(int deviceNumber, ImageDataPtr image)
{
	rawImages[deviceNumber] = image;
}

SharedImageBuffer::ImageDataPtr SharedImageBuffer::getRawImage(int deviceNumber)
{
	return rawImages[deviceNumber];
}

void SharedImageBuffer::removeByDeviceNumber(int deviceNumber)
{//
	// Also remove from syncSet (if present)
	//
	mutex.lock();
	if (syncSet.contains(deviceNumber))
	{
		syncSet.remove(deviceNumber);
		wc.wakeAll();
	}
	mutex.unlock();

	videoMutex.lock();
	videoWCondition.wakeAll();
	videoMutex.unlock();

	audioMutex.lock();
	audioWCondition.wakeAll();
	audioMutex.unlock();
}

void SharedImageBuffer::sync(int deviceNumber)
{
	//
	// Only perform sync if enabled for specified device/stream
	//
	//std::cout << "Waiting count " << nArrived << std::endl;
	mutex.lock();

	if (syncSet.contains(deviceNumber))
	{
		//
		// Increment arrived count
		//
		nArrived++;

		//
		// We are the last to arrive: wake all waiting threads
		//

		if (doSync && (nArrived == syncSet.size()))
			wc.wakeAll();
		//
		// Still waiting for other streams to arrive: wait
		//
		else
		{
			//std::cout << deviceNumber << " Still Waiting count unlock " << nArrived << " Size " << syncSet.size() << " " << doSync << std::endl;
			wc.wait(&mutex);
		}

		//
		// Decrement arrived count
		//
		nArrived--;
	}
	mutex.unlock();
}

void SharedImageBuffer::wakeAll()
{
	QMutexLocker locker(&mutex);
	wc.wakeAll();
}

void SharedImageBuffer::setSyncEnabled(bool enable)
{
	doSync = enable;
}

void SharedImageBuffer::setViewSync(int deviceNumber, bool enable)
{
	if (enable)
	{
		if (!syncSet.contains(deviceNumber))
			syncSet.insert(deviceNumber);
	}
	else
	{
		if (syncSet.contains(deviceNumber))
			syncSet.remove(deviceNumber);
	}
}

bool SharedImageBuffer::isSyncEnabledForDeviceNumber(int deviceNumber)
{
	return syncSet.contains(deviceNumber);
}

bool SharedImageBuffer::getSyncEnabled()
{
	return doSync;
}

int	SharedImageBuffer::getSyncedCameraCount()
{
	return syncSet.size();
}

int SharedImageBuffer::getFirstAvailableViewId()
{
	for (int i = 0; i < m_globalAnimSettings->cameraSettingsList().size(); i++)
	{
		if (isSyncEnabledForDeviceNumber(i))
		{
			return i;
		}
	}
	return -1;
}

void SharedImageBuffer::setStitcher(std::shared_ptr< D360Stitcher > stitcher)
{
	m_stitcher = stitcher;
	//m_stitcher->setGlobalAnimSettings(m_globalAnimSettings);
}

void SharedImageBuffer::setPlaybackStitcher(std::shared_ptr< PlaybackStitcher > stitcher)
{
	m_playbackStitcher = stitcher;
	//m_stitcher->setGlobalAnimSettings(m_globalAnimSettings);
}

bool SharedImageBuffer::syncForVideoProcessing(int videoFrameId)
{
	bool ret = true;
	videoMutex.lock();
	if (videoFrameId > videoProcessedId + 1)
	{
		videoWCondition.wait(&videoMutex, 10);
		ret = false;
	}
	videoMutex.unlock();
	return ret;
}

bool SharedImageBuffer::syncForVideoPlayback(int videoFrameId)
{
	bool ret = true;
	videoMutex.lock();
	if (videoFrameId > videoPlaybackId + 1)
	{
		videoWCondition.wait(&videoMutex, 10);
		ret = false;
	}
	videoMutex.unlock();
	return ret;
}

void SharedImageBuffer::syncForAudioProcessing(int audioFrameId)
{
	audioMutex.lock();
#if 0
	if (audioCapturedCount > audioProcessedId + 1)
#else
	if (audioFrameId > audioProcessedId + 1)
#endif
		audioWCondition.wait(&audioMutex, 1);
	audioMutex.unlock();
}

void SharedImageBuffer::syncForAudioPlayback(int audioFrameId)
{
	audioMutex.lock();
#if 0
	if (audioCapturedCount > audioProcessedId + 1)
#else
	if (audioFrameId > audioPlaybackId + 1)
#endif
		audioWCondition.wait(&audioMutex, 1);
	audioMutex.unlock();
}

void SharedImageBuffer::wakeForVideoProcessing(int videoFrameId)
{
	videoMutex.lock();
	videoProcessedId = videoFrameId;
	videoWCondition.wakeAll();
	videoMutex.unlock();
}

void SharedImageBuffer::wakeForVideoPlayback(int videoFrameId)
{
	videoMutex.lock();
	videoPlaybackId = videoFrameId;
	videoWCondition.wakeAll();
	videoMutex.unlock();
}

void SharedImageBuffer::wakeForAudioProcessing(int audioFrameId)
{
	audioMutex.lock();
	audioProcessedId = audioFrameId;
	audioWCondition.wakeAll();
	audioMutex.unlock();
}

void SharedImageBuffer::wakeForAudioPlayback(int audioFrameId)
{
	audioMutex.lock();
	audioPlaybackId = audioFrameId;
	audioWCondition.wakeAll();
	audioMutex.unlock();
}

void SharedImageBuffer::waitStitcher()
{
	stitcherMutex.lock();
	stitcherWC.wait(&stitcherMutex, 1);
	stitcherMutex.unlock();
}

void SharedImageBuffer::wakeStitcher()
{
	stitcherMutex.lock();
	stitcherWC.wakeOne();
	stitcherMutex.unlock();
}

void SharedImageBuffer::setCaptureFinalizing()
{
	isFinalizing = true;
}

bool SharedImageBuffer::isCaptureFinalizing()
{
	return isFinalizing;
}

void SharedImageBuffer::lockPlaybackBuffer()
{
	playbackBufferMutex.lock();
}

void SharedImageBuffer::unlockPlaybackBuffer()
{
	playbackBufferMutex.unlock();
}

void SharedImageBuffer::addCamera(int cameraIndex)
{
	QMutex *mutex = new QMutex();
	incomingBufferMutex[cameraIndex] = mutex;
}

void SharedImageBuffer::removeCamera(int cameraIndex)
{
	QMutex *mutex = incomingBufferMutex[cameraIndex];
	incomingBufferMutex.remove(cameraIndex);
	delete mutex;
}

void SharedImageBuffer::lockIncomingBuffer(int cameraIndex)
{
	incomingBufferMutex[cameraIndex]->lock();
}

void SharedImageBuffer::unlockIncomingBuffer(int cameraIndex)
{
	incomingBufferMutex[cameraIndex]->unlock();
}

void SharedImageBuffer::lockOculus()
{
	oculusMutex.lock();
}

void SharedImageBuffer::unlockOculus()
{
	oculusMutex.unlock();
}

void SharedImageBuffer::setLiveGrabber(int camIndex)
{
	liveGrabIndex = camIndex;
}

int SharedImageBuffer::getLiveGrabber()
{
	return liveGrabIndex;
}

void SharedImageBuffer::setWeightMapEditEnabled(bool isEnabled)
{
	if (weightEditEnabled != isEnabled)
	{
		weightEditEnabled = isEnabled;
		emit fireEventWeightMapEditEnabled(isEnabled);
	}
}

bool SharedImageBuffer::isWeightMapEditEnabled()
{
	return weightEditEnabled;
}

void SharedImageBuffer::selectView(int viewIndex1, int viewIndex2)
{
	if (isWeightMapEditEnabled())
	{
		emit fireEventViewSelected(viewIndex1, viewIndex2);
	}
	else
	{
		if (selectedViewIndex1 != viewIndex1)
		{
			selectedViewIndex1 = viewIndex1;
			selectedViewIndex2 = viewIndex2;
			emit fireEventViewSelected(viewIndex1, viewIndex2);
		}
	}
}

int SharedImageBuffer::getSelectedView1()
{
	if (isWeightMapEditEnabled())
		return 0;

	return selectedViewIndex1;
}

int SharedImageBuffer::getSelectedView2()
{
	if (isWeightMapEditEnabled())
		return 0;

	return selectedViewIndex2;
}

