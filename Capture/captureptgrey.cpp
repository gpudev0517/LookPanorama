#include "CapturePtGrey.h"
#include <QDebug>
#include "define.h"
#include "CaptureProp.h"
#include "QmlMainWindow.h"

extern QmlMainWindow* g_mainWindow;

void CapturePtGrey::init()
{
    m_pCamera = NULL;
	m_pImage = NULL;
	m_sharedImageBuffer = NULL;

	if (!g_system)
		g_system = System::GetInstance();
	g_nReferred++;
}

void CapturePtGrey::reset(ImageBufferData& frame)
{

}

extern QString getStringFromWString(wstring wstr);
bool CapturePtGrey::open(int index, QString name, int nWidth, int nHeight, Capture::CaptureDomain captureType)
{
    m_nCameraIndex = index;
    m_captureDomain = captureType;

    int dupIndex = 0;
	cout << "naem : " << name.toStdString() << endl;
    //QString uniqueId = g_mainWindow->getDeviceNameByDevicePath(name, dupIndex);

    
	CameraPtr pCam = NULL;
	CameraList camList = g_system->GetCameras();
    unsigned int numCameras = camList.GetSize();
    for (unsigned int i = 0; i < numCameras; i++)
    {
        pCam = camList.GetByIndex(i);
		//cout << "pCam->GetUniqueID() : " << pCam->GetUniqueID() << endl;
		//cout << "uniqueId : " << uniqueId.toStdString() << endl;
        if (getStringFromWString(pCam->GetUniqueID().w_str().c_str()) == name) {
			m_pCamera = camList.GetByIndex(i);
            break;
        }
    }

    if (m_pCamera == NULL)
        return false;

    // Initialize camera
    try
    {
        m_pCamera->Init();/*
		CIntegerPtr width = &m_pCamera->Width;
		width->SetValue(nWidth);
		CIntegerPtr height = &m_pCamera->Height;
		height->SetValue(nHeight);*/
		m_pCamera->PixelFormat.SetValue(PixelFormatEnums::PixelFormat_BayerRG8);
		m_pCamera->BeginAcquisition();
    }
    catch (Spinnaker::Exception &e)
    {
        cout << "Error: " << e.what() << endl;
        //result = -1;
    }

    return true;
}

bool CapturePtGrey::grabFrame(ImageBufferData& frame)
{
	try
	{
		m_pImage = m_pCamera->GetNextImage();
		//ImagePtr pNextResultImage = pCurResultImage;
		/*while (!pNextResultImage->IsIncomplete())
		{
		pCurResultImage = pNextResultImage;
		pNextResultImage = m_pCamera->GetNextImage();
		}*/
		if (m_pImage->IsIncomplete())
		{
			// Retreive and print the image status description
			cout << "Image incomplete: "
				<< Image::GetImageStatusDescription(m_pImage->GetImageStatus())
				<< "..." << endl << endl;
			m_pImage->Release();
			m_pImage = NULL;
			return false;
		}
		else
		{
			//pCurResultImage = pNextResultImage;
			//pNextResultImage = m_pCamera->GetNextImage();
		}

		m_incomingType = D360::Capture::Video;
		return true;
	}
	catch (const std::exception& e)
	{
		cout << "Error: " << e.what() << endl;
		return false;
	}
}

bool CapturePtGrey::retrieveFrame(int channel, ImageBufferData& frame)
{
	if (m_pImage == NULL)
	{
		return false;
	}
	//
	// Print image information; height and width recorded in pixels
	//
	// *** NOTES ***
	// Images have quite a bit of available metadata including
	// things such as CRC, image status, and offset values, to
	// name a few.
	//
	size_t width = m_pImage->GetWidth();
	size_t height = m_pImage->GetHeight();

	//cout << "Grabbed image " << imageCnt << ", width = " << width << ", height = " << height << endl;
	
#ifdef _DEBUG
	//qDebug() << "Current Frame: " << sarg;
#endif

    frame.mFormat = ImageBufferData::BAYERRG8;
    frame.mImageY.buffer = (unsigned char *)m_pImage->GetData();
    frame.mImageY.width = m_pImage->GetWidth();
    frame.mImageY.height = m_pImage->GetHeight();
    frame.mImageY.stride = m_pImage->GetStride();
    frame.mImageU = AlignedImage();
    frame.mImageV = AlignedImage();

	m_pImage->Release();
	m_pImage = NULL;
     m_incomingType = IncomingFrameType::None;
    return true;
}

void CapturePtGrey::close()
{
	if (m_pCamera) {
		try
		{
			m_pCamera->EndAcquisition();
			m_pCamera->DeInit();
		}
		catch (Spinnaker::Exception &e)
		{
			cout << "Error: " << e.what() << endl;
			//result = -1;
		}
		m_pCamera = NULL;
	}
	g_nReferred--;
	if (g_system && (g_nReferred <= 0)) {
		g_system->ReleaseInstance();
		g_system = NULL;
	}
}

double CapturePtGrey::getProperty(int property_id)
{
    if (m_pCamera == NULL)
        return 0.0f;
	switch (property_id)
    {
		case CV_CAP_PROP_FRAME_WIDTH:
		{
			CIntegerPtr width = &m_pCamera->Width;
			if (IsAvailable(width) && IsReadable(width))
				return width->GetValue();
			return 0.0f;
		}
		case CV_CAP_PROP_FRAME_HEIGHT:
		{
			CIntegerPtr height = &m_pCamera->Height;
			if (IsAvailable(height) && IsReadable(height))
				return height->GetValue();
			return 0.0f;
		}
			
    }

    return 0.0f;
}
