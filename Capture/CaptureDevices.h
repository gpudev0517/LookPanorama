#ifndef CAPTUREDEVICES_H
#define CAPTUREDEVICES_H

#include <windows.h>
#include <dshow.h>

#include <vector>
#include <string>

#include <qstring.h>
#include <qlist.h>
#include <qdebug.h>
#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;

#pragma comment(lib, "strmiids")

using namespace std;

struct DeviceMetaInformation
{
    int type;
    string name;
    string path;
};

class CaptureDevices
{
public:
	CaptureDevices();
	virtual ~CaptureDevices();

	HRESULT Enumerate();

    HRESULT GetVideoDevices(vector<DeviceMetaInformation> *videoDevices);
	HRESULT GetAudioDevices(vector<wstring> *audioDevices);

	void getVideoDeviceInfo(std::string devicePath, int &width, int &height, int &fps);
	IMoniker* CaptureDevices::getDeviceMonikerFromPath(std::string devicePath);

	QString getDeviceName();

	std::string getLastError();

private:
	std::string lastError;

	IEnumMoniker *enumMonikerVideo, *enumMonikerAudio;
	IEnumMoniker *enumMonikerVideo_;
};

#endif //CAPTUREDEVICES_H
