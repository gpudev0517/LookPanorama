#include "CaptureDevices.h"
#include "Capture.h"

#include <qstring.h>
#include <qlist.h>
#include <qdebug.h>
#include <libavformat/avformat.h>

#include <iostream>
#include "3DMath.h"

CaptureDevices::CaptureDevices()
{
	CoInitialize(0);
	Enumerate();
}


CaptureDevices::~CaptureDevices()
{
	if (enumMonikerAudio)
		enumMonikerAudio->Release();
	
	if (enumMonikerVideo)
		enumMonikerVideo->Release();

	if (enumMonikerVideo_)
	{
		enumMonikerVideo_->Release();
	}
	
	CoUninitialize();
}

HRESULT CaptureDevices::Enumerate()
{
	HRESULT hr = S_OK;

	ICreateDevEnum *enumDev;

	hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&enumDev));

	if(FAILED(hr))
	{
		lastError = "Could not create device enumerator";
		return hr;
	}

	hr = enumDev->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMonikerVideo, NULL);

	if (FAILED(hr))
	{
		printf("No video capture devices available");
	}

	hr = enumDev->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enumMonikerVideo_, NULL);

	if (FAILED(hr))
	{
		printf("No video capture devices available");
	}

	hr = enumDev->CreateClassEnumerator(CLSID_AudioInputDeviceCategory, &enumMonikerAudio, NULL);

	if (FAILED(hr))
	{
		printf("No audio capture devices available");
	}

	enumDev->Release();

	return hr;
}

extern QString getStringFromWString(wstring wstr);
HRESULT CaptureDevices::GetVideoDevices(vector<DeviceMetaInformation> *deviceList)
{
    // for ffmpeg
    /*if (enumMonikerVideo)
    {
        IMoniker *pMoniker = NULL;

        while (enumMonikerVideo->Next(1, &pMoniker, NULL) == S_OK)
        {
            IPropertyBag *pPropBag;
            HRESULT hr = pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag));
            if (FAILED(hr))
            {
                pMoniker->Release();
                continue;
            }

            VARIANT var;
            wstring friendlyName;
            wstring deviceName;

            VariantInit(&var);
            hr = pPropBag->Read(L"FriendlyName", &var, 0);
            if (SUCCEEDED(hr))
            {
                friendlyName = var.bstrVal;
                VariantClear(&var);
            }

            VariantInit(&var);
            hr = pPropBag->Read(L"DevicePath", &var, 0);
            if (SUCCEEDED(hr))
            {
                deviceName = var.bstrVal;
                VariantClear(&var);
            }

            pPropBag->Release();
            pMoniker->Release();

            if (!friendlyName.empty() && !deviceName.empty())
            {
                DeviceMetaInformation deviceInfo;
                deviceInfo.type = D360::Capture::CAPTURE_DSHOW;
                deviceInfo.name = getStringFromWString(friendlyName).toStdString();
                deviceInfo.path = getStringFromWString(deviceName).toStdString();
                deviceList->push_back(deviceInfo);
            }
        }
    }*/

    // for PtGrey
    {
        SystemPtr system = System::GetInstance();
		CameraList camList = system->GetCameras();
        unsigned int numCameras = camList.GetSize();
		CameraPtr pCam = NULL;
        for (unsigned int i = 0; i < numCameras; i++)
        {
            // Select camera
            pCam = camList.GetByIndex(i);
            DeviceMetaInformation deviceInfo;
            deviceInfo.type = D360::Capture::CAPTURE_PTGREY;
			INodeMap &nodeMapTLDevice = pCam->GetTLDeviceNodeMap();
			CStringPtr ptrDeviceModelName = nodeMapTLDevice.GetNode("DeviceModelName");
			if (IsAvailable(ptrDeviceModelName) && IsReadable(ptrDeviceModelName))
			{
				gcstring deviceModelName = ptrDeviceModelName->ToString();
				cout << deviceModelName << endl << endl;
				deviceInfo.name = getStringFromWString(deviceModelName.w_str().c_str()).toStdString();

			}
			else
				deviceInfo.name = "unknown";

			/*CStringPtr ptrDeviceVendorName = nodeMapTLDevice.GetNode("DeviceVendorName");
			if (IsAvailable(ptrDeviceVendorName) && IsReadable(ptrDeviceVendorName))
			{
				gcstring deviceVendorName = ptrDeviceVendorName->ToString();
				cout << deviceVendorName << " ";
			}*/
            
            deviceInfo.path = getStringFromWString(pCam->GetUniqueID().w_str().c_str()).toStdString();
            deviceList->push_back(deviceInfo);
        }
        pCam = NULL;
        camList.Clear();
        system->ReleaseInstance();
    }

	return S_OK;
}

HRESULT GetPin(IBaseFilter *pFilter, PIN_DIRECTION PinDir, IPin **ppPin)
{
	IEnumPins  *pEnum = NULL;
	IPin       *pPin = NULL;
	HRESULT    hr;

	if (ppPin == NULL)
	{
		return E_POINTER;
	}

	hr = pFilter->EnumPins(&pEnum);
	if (FAILED(hr))
	{
		return hr;
	}
	while (pEnum->Next(1, &pPin, 0) == S_OK)
	{
		PIN_DIRECTION PinDirThis;
		hr = pPin->QueryDirection(&PinDirThis);
		if (FAILED(hr))
		{
			pPin->Release();
			pEnum->Release();
			return hr;
		}
		if (PinDir == PinDirThis)
		{
			// Found a match. Return the IPin pointer to the caller.
			*ppPin = pPin;
			pEnum->Release();
			return S_OK;
		}
		// Release the pin for the next time through the loop.
		pPin->Release();
	}
	// No more pins. We did not find a match.
	pEnum->Release();
	return E_FAIL;
}

// Given a pin, find a preferred media type 
//
// pPin         Pointer to the pin.
// ppmt         Receives a pointer to the media type. Can be NULL.
//
// Note: If you want to check whether a pin supports a desired media type,
//       but do not need the format details, set ppmt to NULL.
//
//       If ppmt is not NULL and the method succeeds, the caller must
//       delete the media type, including the format block. 

HRESULT GetPinMediaType(
	IPin *pPin,             // pointer to the pin
	int& width, int& height, int& fps
	)
{
	AM_MEDIA_TYPE **ppmt = NULL;

	IEnumMediaTypes *pEnum = NULL;
	AM_MEDIA_TYPE *pmt = NULL;
	BOOL bFound = FALSE;
	VIDEOINFOHEADER* videoInfoHeader = NULL;

	fps = 0;

	HRESULT hr = pPin->EnumMediaTypes(&pEnum);
	if (FAILED(hr))
	{
		return hr;
	}

	while (hr = pEnum->Next(1, &pmt, NULL), hr == S_OK)
	{
		// Found a match. 
		if ((pmt->formattype == FORMAT_VideoInfo) &&
			(pmt->cbFormat >= sizeof(VIDEOINFOHEADER)) &&
			(pmt->pbFormat != NULL))
		{
			videoInfoHeader = (VIDEOINFOHEADER*)pmt->pbFormat;
			
			int w = videoInfoHeader->bmiHeader.biWidth;  // Supported width
			int h = videoInfoHeader->bmiHeader.biHeight; // Supported height
			int f = 10000000 / videoInfoHeader->AvgTimePerFrame;

			// Get highest resolution, at the top speed
			if (fps < f)
			{
				width = w;
				height = h;
				fps = f;
			}
			else if (fps == f)
			{
				if (width * height < w * h)
				{
					width = w;
					height = h;
				}
			}
			bFound = TRUE;
		}		
	}

	if (&pEnum)
	{
		pEnum->Release();
		pEnum = NULL;
	}
	if (SUCCEEDED(hr))
	{
		if (!bFound)
		{
			hr = VFW_E_NOT_FOUND;
		}
	}
	return hr;
}


void CaptureDevices::getVideoDeviceInfo(std::string devicePath, int &width, int &height, int &fps)
{
	// get pMonitor from devicePath
	IMoniker* pMoniker = getDeviceMonikerFromPath(devicePath);
	if (pMoniker == NULL)
	{
		// Exception: Device not found
		return;
	}

	//To create a DirectShow capture filter for the device, call the IMoniker::BindToObject method to get an IBaseFilter pointer.

	// bind the device moniker to a base filter
	IBaseFilter *pCap = NULL;
	HRESULT hr = pMoniker->BindToObject(0, 0, IID_IBaseFilter, (void**)&pCap);
	pMoniker->Release();

	// get an output pin from that filter
	IPin* pin = NULL;
	GetPin(pCap, PIN_DIRECTION::PINDIR_OUTPUT, &pin);
	
	// enumerate over media types of that output pin
	//int width, height, fps;
	GetPinMediaType(pin, width, height, fps);
}

IMoniker* CaptureDevices::getDeviceMonikerFromPath(std::string devicePath)
{
	if (!enumMonikerVideo_)
		return NULL;

	IMoniker *pFoundMoniker = NULL;
	IMoniker *pMoniker = NULL;

	while (enumMonikerVideo_->Next(1, &pMoniker, NULL) == S_OK)
	{
		IPropertyBag *pPropBag;
		HRESULT hr = pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag));
		if (FAILED(hr))
		{
			pMoniker->Release();
			continue;
		}

		VARIANT var;
		wstring friendlyName;
		wstring deviceName;
		
		VariantInit(&var);
		hr = pPropBag->Read(L"DevicePath", &var, 0);
		if (SUCCEEDED(hr))
		{
			deviceName = var.bstrVal;
			VariantClear(&var);

			QString dPath = QString::fromWCharArray(var.bstrVal);

			if (dPath == devicePath.c_str())
			{
				pFoundMoniker = pMoniker;
			}
		}

		pPropBag->Release();

		if (pFoundMoniker)
			break;
		pMoniker->Release();
	}

	return pFoundMoniker;
}

HRESULT CaptureDevices::GetAudioDevices(vector<wstring> *audioDevices)
{
	if (!enumMonikerAudio)
		return E_FAIL;

	IMoniker *pMoniker = NULL;
	wstring name;

	while (enumMonikerAudio->Next(1, &pMoniker, NULL) == S_OK)
	{
		IPropertyBag *pPropBag;
		HRESULT hr = pMoniker->BindToStorage(0, 0, IID_PPV_ARGS(&pPropBag));
		if (FAILED(hr))
		{
			pMoniker->Release();
			continue;  
		} 

		VARIANT var;
		VariantInit(&var);

		hr = pPropBag->Read(L"FriendlyName", &var, 0);
		if (SUCCEEDED(hr))
		{
			name = var.bstrVal;
			VariantClear(&var); 
		}

		pPropBag->Release();
		pMoniker->Release();

		if (!name.empty())
			audioDevices->push_back(name);
	}

	return S_OK;
}
