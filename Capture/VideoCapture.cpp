#include "VideoCapture.h"

//VIDEODEVICE
//public
VideoDevice::VideoDevice()
{
	friendlyname = (char*)calloc(1, MAX_DEVICE_NAME * sizeof(char));
	filtername = (WCHAR*)calloc(1, MAX_DEVICE_NAME * sizeof(WCHAR));
	filter = 0;
}

VideoDevice::~VideoDevice()
{
	free(friendlyname);
	free(filtername);
}

const char* VideoDevice::GetFriendlyName()
{
	return friendlyname;
}

// ROT

HRESULT AddToRot(IUnknown *pUnkGraph, DWORD *pdwRegister)
{
	IMoniker * pMoniker = NULL;
	IRunningObjectTable *pROT = NULL;

	if (FAILED(GetRunningObjectTable(0, &pROT)))
	{
		return E_FAIL;
	}

	const size_t STRING_LENGTH = 256;

	WCHAR wsz[STRING_LENGTH];

	StringCchPrintfW(
		wsz, STRING_LENGTH,
		L"FilterGraph %08x pid %08x",
		(DWORD_PTR)pUnkGraph,
		GetCurrentProcessId()
		);

	HRESULT hr = CreateItemMoniker(L"!", wsz, &pMoniker);
	if (SUCCEEDED(hr))
	{
		hr = pROT->Register(ROTFLAGS_REGISTRATIONKEEPSALIVE, pUnkGraph,
			pMoniker, pdwRegister);
		pMoniker->Release();
	}
	pROT->Release();

	return hr;
}

void RemoveFromRot(DWORD pdwRegister)
{
	IRunningObjectTable *pROT;
	if (SUCCEEDED(GetRunningObjectTable(0, &pROT))) {
		pROT->Revoke(pdwRegister);
		pROT->Release();
	}
}

HRESULT SaveGraphFile(IGraphBuilder *pGraph, WCHAR *wszPath)
{
	const WCHAR wszStreamName[] = L"ActiveMovieGraph";
	HRESULT hr;

	IStorage *pStorage = NULL;
	hr = StgCreateDocfile(
		wszPath,
		STGM_CREATE | STGM_TRANSACTED | STGM_READWRITE | STGM_SHARE_EXCLUSIVE,
		0, &pStorage);
	if (FAILED(hr))
	{
		return hr;
	}

	IStream *pStream;
	hr = pStorage->CreateStream(
		wszStreamName,
		STGM_WRITE | STGM_CREATE | STGM_SHARE_EXCLUSIVE,
		0, 0, &pStream);
	if (FAILED(hr))
	{
		pStorage->Release();
		return hr;
	}

	IPersistStream *pPersist = NULL;
	pGraph->QueryInterface(IID_IPersistStream, (void**)&pPersist);
	hr = pPersist->Save(pStream, TRUE);
	pStream->Release();
	pPersist->Release();
	if (SUCCEEDED(hr))
	{
		hr = pStorage->Commit(STGC_DEFAULT);
	}
	pStorage->Release();
	return hr;
}

//VIDEOCAPTURE
//public
VideoCapture::VideoCapture(VideoCaptureCallback cb, int index, const char* cameraName, float fps)
{
	CoInitialize(NULL);

	playing = false;
	current = 0;
	callbackhandler = new CallbackHandler(cb, index);
	device = new VideoDevice();
	this->fps = fps;

	InitializeGraph();
	SetSourceFilters(cameraName);
	SetSampleGrabber();
	SetNullRenderer();

	Select(device);

	//SaveGraphFile((IGraphBuilder*)capture, L"E:/MyGraph.grf");
}

VideoCapture::~VideoCapture()
{
	RemoveFromRot(dwRegister);
	delete callbackhandler;
	delete device;
}

void VideoCapture::Select(VideoDevice* dev)
{
	HRESULT hr;
	LONGLONG start = MAXLONGLONG, stop = MAXLONGLONG;
	bool was_playing = playing;

	if (!dev->filter) throw E_INVALIDARG;

	//temporary stop
	if (playing) Stop();

	if (current)
	{
		//remove and add the filters to force disconnect of pins
		graph->RemoveFilter(current->filter);
		graph->RemoveFilter(samplegrabberfilter);

		graph->AddFilter(samplegrabberfilter, L"Sample Grabber");
		graph->AddFilter(current->filter, current->filtername);
	}

	start = 0;
	current = dev;

	//connect graph with current source filter
#ifdef SHOW_DEBUG_RENDERER
	hr = capture->RenderStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, current->filter, samplegrabberfilter, NULL);
#else
	hr = capture->RenderStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, current->filter, samplegrabberfilter, nullrenderer);
#endif
	if (hr != S_OK) throw hr;

	// fps change
	IAMStreamConfig *pVSC;
	hr = capture->FindInterface(&PIN_CATEGORY_CAPTURE,
		&MEDIATYPE_Video,
		current->filter, IID_IAMStreamConfig, (void **)&pVSC);
	AM_MEDIA_TYPE *pmt = 0;
	hr = pVSC->GetFormat(&pmt);

	// DV capture does not use a VIDEOINFOHEADER
	if (hr == NOERROR)
	{
		if (pmt->formattype == FORMAT_VideoInfo)
		{
			VIDEOINFOHEADER *pvi = (VIDEOINFOHEADER *)pmt->pbFormat;
			pvi->AvgTimePerFrame = (LONGLONG)(10000000 / fps);
			hr = pVSC->SetFormat(pmt);
		}
		//DeleteMediaType(pmt);
	}

	//start streaming
	hr = capture->ControlStream(&PIN_CATEGORY_CAPTURE, &MEDIATYPE_Video, current->filter, &start, &stop, 1, 2);
	if (hr < 0) throw hr;

	//restart
	if (was_playing) Start();
}

void VideoCapture::Start()
{
	HRESULT hr;

	hr = control->Run();
	if (hr < 0) throw hr;

	playing = true;
}

void VideoCapture::Stop()
{
	HRESULT hr;

	hr = control->StopWhenReady();
	if (hr < 0) throw hr;

	playing = false;
}

int VideoCapture::getWidth()
{
	AM_MEDIA_TYPE mt;
	HRESULT hr = samplegrabber->GetConnectedMediaType(&mt);
	if (FAILED(hr)) {
		return -1;
	}

	VIDEOINFOHEADER *pVih = (VIDEOINFOHEADER *)mt.pbFormat;
	return pVih->bmiHeader.biWidth;
}

int VideoCapture::getHeight()
{
	AM_MEDIA_TYPE mt;
	HRESULT hr = samplegrabber->GetConnectedMediaType(&mt);
	if (FAILED(hr)) {
		return -1;
	}

	VIDEOINFOHEADER *pVih = (VIDEOINFOHEADER *)mt.pbFormat;
	return pVih->bmiHeader.biHeight;
}

//protected
void VideoCapture::InitializeGraph()
{
	HRESULT hr;

	//create the FilterGraph
	hr = CoCreateInstance(CLSID_FilterGraph, NULL, CLSCTX_INPROC_SERVER, IID_IFilterGraph2, (void**)&graph);
	if (hr < 0) throw hr;

	//create the CaptureGraphBuilder
	hr = CoCreateInstance(CLSID_CaptureGraphBuilder2, NULL, CLSCTX_INPROC_SERVER, IID_ICaptureGraphBuilder2, (void**)&capture);
	if (hr < 0) throw hr;

	AddToRot(capture, &dwRegister);

	//get the controller for the graph
	hr = graph->QueryInterface(IID_IMediaControl, (void**)&control);
	if (hr < 0) throw hr;

	capture->SetFiltergraph(graph);
}

void VideoCapture::SetSourceFilters(const char* camName)
{
	HRESULT hr;
	VARIANT name;
	LONGLONG start = MAXLONGLONG, stop = MAXLONGLONG;

	unsigned long dev_count = 0;

	ICreateDevEnum*		dev_enum;
	IEnumMoniker*		enum_moniker;
	IMoniker*			moniker;
	IPropertyBag*		pbag;

	//create an enumerator for video input devices
	hr = CoCreateInstance(CLSID_SystemDeviceEnum, NULL, CLSCTX_INPROC_SERVER, IID_ICreateDevEnum, (void**)&dev_enum);
	if (hr < 0) throw hr;

	hr = dev_enum->CreateClassEnumerator(CLSID_VideoInputDeviceCategory, &enum_moniker, NULL);
	if (hr < 0) throw hr;
	if (hr == S_FALSE) return; //no devices found

	//get devices (max 8)
	ULONG cFetched = 0;
	unsigned int i = 0;
	while (enum_moniker->Next(1, &moniker, &cFetched) == S_OK)
		//enum_moniker->Next(MAX_DEVICES, &moniker, &dev_count);
		//for (unsigned int i=0; i<dev_count; i++)
	{
		bool isSelected = false;
		dev_count++;
		//get properties
		hr = moniker->BindToStorage(0, 0, IID_IPropertyBag, (void**)&pbag);
		if (hr >= 0)
		{
			VariantInit(&name);

			//get the description
			hr = pbag->Read(L"Description", &name, 0);
			if (hr < 0) hr = pbag->Read(L"FriendlyName", &name, 0);
			if (hr >= 0)
			{
				//Initialize the VideoDevice struct
				VideoDevice* dev = device;
				BSTR ptr = name.bstrVal;

				for (int c = 0; *ptr; c++, ptr++)
				{
					//bit hacky, but i don't like to include ATL
					dev->filtername[c] = *ptr;
					dev->friendlyname[c] = *ptr & 0xFF;
				}

				// select corresponding cameras
				if (strcmp(camName, dev->friendlyname) != 0) continue;

				//add a filter for the device
				hr = graph->AddSourceFilterForMoniker(moniker, 0, dev->filtername, &dev->filter);
				isSelected = true;
			}
			VariantClear(&name);
			pbag->Release();
		}
		moniker->Release();

		i++;

		if (isSelected) break;
	}
}

void VideoCapture::SetSampleGrabber()
{
	HRESULT hr;

	hr = CoCreateInstance(CLSID_SampleGrabber, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)&samplegrabberfilter);
	if (hr < 0) throw hr;

	hr = graph->AddFilter(samplegrabberfilter, L"Sample Grabber");
	if (hr != S_OK) throw hr;

	hr = samplegrabberfilter->QueryInterface(IID_ISampleGrabber, (void**)&samplegrabber);
	if (hr != S_OK) throw hr;

	//set the media type
	AM_MEDIA_TYPE mt;
	memset(&mt, 0, sizeof(AM_MEDIA_TYPE));

	mt.majortype = MEDIATYPE_Video;
	mt.subtype = MEDIASUBTYPE_RGB24;
	// setting the above to 32 bits fails consecutive Select for some reason
	// and only sends one single callback (flush from previous one ???)
	// must be deeper problem. 24 bpp seems to work fine for now.

	callbackhandler->SetMediaType(&mt);
	hr = samplegrabber->SetMediaType(&mt);
	if (hr != S_OK) throw hr;

	samplegrabber->SetCallback(callbackhandler, 0);
}

void VideoCapture::SetNullRenderer()
{
	HRESULT hr;

	hr = CoCreateInstance(CLSID_NullRenderer, NULL, CLSCTX_INPROC_SERVER, IID_IBaseFilter, (void**)&nullrenderer);
	if (hr < 0) throw hr;

	graph->AddFilter(nullrenderer, L"Null Renderer");
}



//CALLBACKHANDLER
//public
VideoCapture::CallbackHandler::CallbackHandler(VideoCaptureCallback cb, int index)
{
	callback = cb;
	m_index = index;
}

VideoCapture::CallbackHandler::~CallbackHandler()
{
}

void VideoCapture::CallbackHandler::SetMediaType(AM_MEDIA_TYPE* am)
{
	if (am->subtype == MEDIASUBTYPE_RGB555) bitpixel = 16;
	else if (am->subtype == MEDIASUBTYPE_RGB24) bitpixel = 24;
	else if (am->subtype == MEDIASUBTYPE_RGB32) bitpixel = 32;
}

HRESULT VideoCapture::CallbackHandler::SampleCB(double time, IMediaSample *sample)
{
	HRESULT hr;
	AM_MEDIA_TYPE* mt;
	unsigned char* buffer;

	hr = sample->GetPointer((BYTE**)&buffer);
	if (hr != S_OK) return S_OK;

	hr = sample->GetMediaType(&mt);
	if (hr < 0) return S_OK;
	if (hr == S_OK) SetMediaType(mt);

	callback(m_index, buffer, sample->GetActualDataLength(), bitpixel);
	return S_OK;
}

HRESULT VideoCapture::CallbackHandler::BufferCB(double time, BYTE *buffer, long len)
{
	return S_OK;
}

HRESULT VideoCapture::CallbackHandler::QueryInterface(const IID &iid, LPVOID *ppv)
{
	if (iid == IID_ISampleGrabberCB || iid == IID_IUnknown)
	{
		*ppv = (void *) static_cast<ISampleGrabberCB*>(this);
		return S_OK;
	}
	return E_NOINTERFACE;
}

ULONG VideoCapture::CallbackHandler::AddRef()
{
	return 1;
}

ULONG VideoCapture::CallbackHandler::Release()
{
	return 2;
}
