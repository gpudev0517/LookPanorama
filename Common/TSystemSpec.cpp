/* ====< インクルード >====================================================== */
#if 0
//#include "prefix.h"
//#include "TGL.h"
#include "TSystemTypes.h"
//#include "Sound/Api/cEWSoundApiXA2.h"


//#include "Graphics/Api/cTGLGraphicsDX9.h"
#include <windows.h>
//#include <winuser.h>


#include <dxdiag.h>
#include <setupapi.h>
#include <devguid.h>
//#include "utility/TGLDirectXDialog.h"

#include <QString>

#include <stdint.h>

typedef uint32_t u32;
typedef int32_t s32;
typedef uint16_t u16;

#pragma warning(disable: 4996)

/* ====< 定義 >====================================================== */

//#define	CPUCACHE_DETAIL	//!< CPUキャッシュを詳細に調べる

namespace {
	//!	CPUのベンダーID
	enum CPU_VENDER {
		UNKNOWN = 0,
		INTEL,
		AMD,
		TMx86,
		CYRIX,
		IDT,
		SIS,
		UMC,
		RISE,
		NEXGEN,
		COMPAQ,
		NSC,
	};
	//!	CPUフラグ
	enum CPUF{
		CPUF_MMX				= (1<<0),
		CPUF_MMXEx				= (1<<1),
		CPUF_SSE				= (1<<2),
		CPUF_SSE2				= (1<<3),
		CPUF_SSE3				= (1<<4),
		CPUF_SSSE3				= (1<<5),
		CPUF_SSE4				= (1<<6),
		CPUF_SSE41				= (1<<7),
		CPUF_SSE42				= (1<<8),
		CPUF_SSE4A				= (1<<9),
		CPUF_SSE5				= (1<<10),
		CPUF_AVX				= (1<<11),
		CPUF_3DNow				= (1<<12),
		CPUF_3DNowEx			= (1<<13),
		CPUF_HT					= (1<<14),
		CPUF_VT					= (1<<15),
		CPUF_AmdV				= (1<<16),
		CPUF_AA64				= (1<<17),
		CPUF_IA64				= (1<<18),
		CPUF_SpeedStep			= (1<<19),
		CPUF_EIST				= (1<<20),
		CPUF_PowerNow			= (1<<21),
		CPUF_LongHaul			= (1<<22),
		CPUF_LongRun			= (1<<23),
		CPUF_ClockModulation	= (1<<24),
		CPUF_ProcessorSerial	= (1<<25),
		CPUF_IA32e				= (1<<26),
		CPUF_NX					= (1<<27),
		CPUF_MSR				= (1<<28),
		CPUF_K7Sempron			= (1<<29),
		CPUF_DualCore			= (1<<30),
		CPUF_EistCorrect		= (1<<31),
		//CPUF_K8100MHzSteps	= (1<<32),
		//CPUF_K8Under1100V		= (1<<33),
	};

	#ifdef X64
	extern "C" {
		void __fastcall _cpuid(DWORD dwOP, DWORD *lpAX, DWORD *lpBX, DWORD *lpCX, DWORD *lpDX){}
	}
	#endif // _X86_64


	//=============================================================================
	// x64
	//=============================================================================
	/*
		switch (GetBaseSystemInfo()) {
		case SYSINFO_32PROCESS_ON_WO64:
			printf("WOW64");
			break;
		case SYSINFO_64PROCESS:
			printf("x64");
			break;
		case SYSINFO_32PROCESS:
			printf("x86");
			break;
	*/

	typedef BOOL (WINAPI *LPFN_ISWOW64PROCESS) (HANDLE, PBOOL);
	typedef void (WINAPI *LPFN_GETNATIVESYSTEMINFO) (LPSYSTEM_INFO);

	typedef enum {
		SYSINFO_UNKNOWN = 0,
		SYSINFO_32PROCESS = 1,
		SYSINFO_32PROCESS_ON_WO64 = 2,
		SYSINFO_64PROCESS = 3
	} BaseSystemInfo;

	// ---------------------------------------------
	//! @brief  
	//! @return 
	// ---------------------------------------------
	BOOL Isx64Architecture()
	{
		LPFN_GETNATIVESYSTEMINFO fnGetNativeSystemInfo = NULL;
		BOOL bIsx64 = false;
		SYSTEM_INFO info;

		if (fnGetNativeSystemInfo == NULL) {
			fnGetNativeSystemInfo = (LPFN_GETNATIVESYSTEMINFO) GetProcAddress(
				GetModuleHandle(TEXT("kernel32")), TEXT("GetNativeSystemInfo"));
		}

		if (fnGetNativeSystemInfo != NULL) {
			fnGetNativeSystemInfo(&info);
			if (info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_AMD64) {
				bIsx64 = true;
			} else if (info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_IA64) {
				bIsx64 = true;
			} else if (info.wProcessorArchitecture == PROCESSOR_ARCHITECTURE_INTEL) {
				bIsx64 = false;
			}
		}
		return bIsx64;
	}

	// ---------------------------------------------
	//! @brief  
	//! @return 
	// ---------------------------------------------
	BOOL IsWow64()
	{
		BOOL bIsWow64 = false;
		LPFN_ISWOW64PROCESS fnIsWow64Process = (LPFN_ISWOW64PROCESS)GetProcAddress(GetModuleHandle("kernel32"),"IsWow64Process");
		if (NULL != fnIsWow64Process)
		{
			if (!fnIsWow64Process(GetCurrentProcess(),&bIsWow64))
			{
				// handle error
				bIsWow64 = false;
			}
		}
		return bIsWow64;
	}

	// ---------------------------------------------
	//! @brief  
	//! @return 
	// ---------------------------------------------
	BaseSystemInfo GetBaseSystemInfo()
	{
		if (IsWow64() == false) {
			return SYSINFO_32PROCESS;
		} else if (Isx64Architecture() == TRUE) {
			return SYSINFO_64PROCESS;
		} else {
			return SYSINFO_32PROCESS_ON_WO64;
		}

		return SYSINFO_UNKNOWN;
	}
};

/* ========================================================================= */
/*		スペック出力														 */
/* ========================================================================= */
// ---------------------------------------------
//! @brief  PCスペックをテキストに出力する
//! @return trueなら成功
// ---------------------------------------------
bool outputSpecInfo()
{
	char	str[256];

	//!	ヘッダ
	QString strPrint;
	strPrint += "\n=========================================================\n";
	strPrint += "\tPC Specification\n";
	strPrint += "=========================================================\n";
	printf("%s", strPrint.toStdString());

	// -----------------------------------------
	//	OS
	const char*	OS_NAME[SYSINFOOS_MAX] = {
		"Unknown",
		"Windows",
		"Windows 95",
		"Windows 98",
		"Windows Me",
		"Windows 9x",
		"Windows 3.x",
		"Windows NT 4.0",
		"Windows 2000",
		"Windows XP",
		"Windows Server 2003",
		"Windows Vista",
		"Windows 7",
		"Windows 8",
	};
	OSVERSIONINFOEX	osVerInfo;
	printf("【 OS 】\n");
	
	QString strProduct;

	osVerInfo.dwOSVersionInfoSize = sizeof( OSVERSIONINFOEX );
	if( GetVersionEx( (OSVERSIONINFO*)&osVerInfo ) ){
		osVerInfo.dwOSVersionInfoSize = sizeof(OSVERSIONINFO);
		GetVersionEx((OSVERSIONINFO*)&osVerInfo);
	}
	int mOS = SYSINFOOS_UNKNOWN;
	switch( osVerInfo.dwPlatformId ){
		case VER_PLATFORM_WIN32_WINDOWS:	//	Win98系
			switch( osVerInfo.dwMinorVersion ){
				case 0:		mOS = SYSINFOOS_WINDOWS_95;	break;
				case 10:	mOS = SYSINFOOS_WINDOWS_98;	break;
				case 90:	mOS = SYSINFOOS_WINDOWS_ME;	break;
				default:
					mOS = SYSINFOOS_WINDOWS_9X;
					break;
			}
			break;
		case VER_PLATFORM_WIN32_NT:	//	WinNT系
			switch( osVerInfo.dwMajorVersion ){
				case 3:
					mOS = SYSINFOOS_WINDOWS_3_X;
					break;
				case 4:
					mOS = SYSINFOOS_WINDOWS_NT4;
					break;
				case 5:
					switch( osVerInfo.dwMinorVersion ){
						case 0:	mOS = SYSINFOOS_WINDOWS_2000;	break;
						case 1:	mOS = SYSINFOOS_WINDOWS_XP;		break;
						case 2:	mOS = SYSINFOOS_WINDOWS_S2003;	break;
					}
					break;
				case 6:
					switch( osVerInfo.dwMinorVersion ){
						case 0:	mOS = SYSINFOOS_WINDOWS_VISTA;	break;
						case 1:	mOS = SYSINFOOS_WINDOWS_7;		break;
						case 2: mOS = SYSINFOOS_WINDOWS_8;		break;
						default:
							mOS = SYSINFOOS_WINDOWS;
					}

					{
						BOOL (CALLBACK* pfnGetProductInfo)(DWORD dwOSMajorVersion,DWORD dwOSMinorVersion,DWORD dwSpMajorVersion,DWORD dwSpMinorVersion,PDWORD pdwReturnedProductType);

						HMODULE	hModule = ::LoadLibrary("kernel32.dll");
						if(hModule == NULL)
							return	false;

						(*(FARPROC*)&pfnGetProductInfo) = ::GetProcAddress(hModule,"GetProductInfo");

						// Vistaの場合にはGetProductInfo関数を使用して
						// Edition情報を取得する
						DWORD edition = PRODUCT_UNDEFINED;
						if (pfnGetProductInfo(osVerInfo.dwMajorVersion,osVerInfo.dwMinorVersion,osVerInfo.wServicePackMajor,osVerInfo.wServicePackMinor,&edition))
						{
							switch (edition)
							{
								case PRODUCT_ENTERPRISE:
								case 0x0000001B://PRODUCT_ENTERPRISE_N;
									strProduct = " Enterprise";
									break;
								case PRODUCT_ULTIMATE:
								case 0x0000001C://PRODUCT_ULTIMATE_N:
									strProduct = " Ultimate";
									break;
								case PRODUCT_BUSINESS:
								case PRODUCT_BUSINESS_N:
									strProduct = " Business";
									break;
								case PRODUCT_HOME_PREMIUM:
								case 0x0000001A://PRODUCT_HOME_PREMIUM_N:
									strProduct = " Home Premium";
									break;
								case PRODUCT_HOME_BASIC:
								case PRODUCT_HOME_BASIC_N:
									strProduct = " Home Basic";
									break;
								case 0x0000000B://PRODUCT_STARTER
								case 0x0000002F:
									strProduct = " Starter";
									break;
								case 0x00000030://PRODUCT_PROFESSIONAL:
								case 0x00000031:
									strProduct = " Professional";
									break;
								default:
									//strProduct = " Unknown";
									break;
							}
						}
					}
					break;
				default:
					mOS = SYSINFOOS_WINDOWS;
					break;
			}
			break;
	}
	sprintf( str, "  %s", OS_NAME[mOS] );

	char OSBit[6];
	switch (GetBaseSystemInfo()) {
        case SYSINFO_32PROCESS_ON_WO64:
            sprintf(OSBit,"64");
            break;
        case SYSINFO_64PROCESS:
            sprintf(OSBit,"64");
            break;
        case SYSINFO_32PROCESS:
            sprintf(OSBit,"32");
            break;
		default:
			sprintf(OSBit,"??");
			break;
    }


	printf("%s [%sbit] %s  [%d.%d Build %d] %s\n",
		str,
		OSBit,
		osVerInfo.szCSDVersion,
		//osVerInfo.dwPlatformId,
		osVerInfo.dwMajorVersion,
		osVerInfo.dwMinorVersion,
		osVerInfo.dwBuildNumber,
		strProduct.toStdString());

	//	地域
	DWORD mLocaleId;
	GetLocaleInfo( mLocaleId, LOCALE_SCOUNTRY, str, strlen(str) );
	printf("  LCID:%d(%s)\n", mLocaleId, str );

	// -----------------------------------------
	//	CPU
	outputCPUInfo();

	// -----------------------------------------
	//	メモリ
	MEMORYSTATUSEX	memSta;
	printf( "\n【 Memory 】\n" );
	memSta.dwLength = sizeof( MEMORYSTATUSEX );
	GlobalMemoryStatusEx( &memSta );
	
#if 1
	sprintf( str, "  物理メモリ : %5d /%5d[MB]\n", (s32)BYTE_TO_MB(memSta.ullAvailPhys), (s32)BYTE_TO_MB(memSta.ullTotalPhys) );	TGL::Logger::Print( str );
	sprintf( str, "  ページング : %5d /%5d[MB]\n", (s32)BYTE_TO_MB(memSta.ullAvailPageFile), (s32)BYTE_TO_MB(memSta.ullTotalPageFile) );	TGL::Logger::Print( str );
	sprintf( str, "  仮想メモリ : %5d /%5d[MB]\n", (s32)BYTE_TO_MB(memSta.ullAvailVirtual), (s32)BYTE_TO_MB(memSta.ullTotalVirtual) );	TGL::Logger::Print( str );

	printf("\n\n");
#else
	
	TGL::Logger::Print( "  物理メモリ : %sbyte /%sbyte\n",STR_MB(memSta.ullAvailPhys).c_str(),STR_MB(memSta.ullTotalPhys).c_str() );
	TGL::Logger::Print( "  ページング : %sbyte /%sbyte\n",STR_MB(memSta.ullAvailPageFile).c_str(),STR_MB(memSta.ullTotalPageFile).c_str() );
	TGL::Logger::Print( "  仮想メモリ : %sbyte /%sbyte\n",STR_MB(memSta.ullAvailVirtual).c_str(),STR_MB(memSta.ullTotalVirtual).c_str() );
#endif
	// ドライブ情報
	//outputDriveInfo();

	return true;
}

//-------------------------------------------------------------
// HDEVINFO（SP_DEVINFO_DATA）のプロパティ文字列取得
// SetupDiGetDeviceRegistryPropertyにより文字列データを取得する
//-------------------------------------------------------------
bool	GetDeviceRegistryPropertyString(HDEVINFO hDeviceInfoSet,PSP_DEVINFO_DATA pDeviceInfoData,DWORD dwProperty,QString& strText)
{
	BOOL	ret;
	DWORD	dwDataType;
	DWORD	dwDataSize;
	unsigned char*	pData;

	pData = NULL;
	dwDataSize = 0;
	while(1)
	{
		ret = ::SetupDiGetDeviceRegistryProperty(hDeviceInfoSet,pDeviceInfoData,dwProperty,&dwDataType,pData,dwDataSize,&dwDataSize);
		if(ret || ::GetLastError() != ERROR_INSUFFICIENT_BUFFER)
			break;

		if(pData)
			delete	pData;
		pData = new unsigned char[dwDataSize];
	}
	if(ret && pData)
	{
		//TODO:REG_MULTI_SZも1つの文字列として扱っている！
		if(dwDataType == REG_EXPAND_SZ || dwDataType == REG_MULTI_SZ || dwDataType == REG_SZ)
			strText = (char*)pData;
	}
	if(pData)
		delete	pData;

	return	(ret && pData) ? true : false;
}


//-------------------------------------------------------------
// デバイス列挙
//-------------------------------------------------------------
void enumDeviceName(HDEVINFO hDI)
{
    for(int i=0;;i++){
		SP_DEVINFO_DATA devinfo;
        ZeroMemory(&devinfo, sizeof(devinfo));
        devinfo.cbSize = sizeof(SP_DEVINFO_DATA);
        if(!SetupDiEnumDeviceInfo(hDI, i, &devinfo)){
            break;
        }

		// SPDRP_ENUMRATOR_NAME
		QString strType;
		GetDeviceRegistryPropertyString(hDI, &devinfo, SPDRP_ENUMERATOR_NAME,strType);
		// SPDRP_MFG
		QString strMfg;
		GetDeviceRegistryPropertyString(hDI, &devinfo, SPDRP_MFG,strMfg);

		// SPDRP_FRIENDLYNAME
		QString strFriendlyName;
		GetDeviceRegistryPropertyString(hDI, &devinfo, SPDRP_FRIENDLYNAME,strFriendlyName);
		
		if( strType != "ROOT"){
			printf("  %s[%s] : %s\n", strMfg.toStdString(), strType.toStdString(), strFriendlyName.toStdString());
		}
    }
}

// ---------------------------------------------
//! @brief  ドライブ情報
//! @return true:成功
// ---------------------------------------------
bool outputDriveInfo()
{
	printf( "\n【 Drive 】\n" );

	{	//!	HDDドライブ
		HDEVINFO hDI = SetupDiGetClassDevs((LPGUID) &GUID_DEVCLASS_DISKDRIVE, NULL,0,DIGCF_PRESENT );
	    if( INVALID_HANDLE_VALUE != hDI )
			enumDeviceName(hDI);
	    SetupDiDestroyDeviceInfoList(hDI);
	}

	{	//!	CDドライブ
		HDEVINFO hDI = SetupDiGetClassDevs((LPGUID) &GUID_DEVCLASS_CDROM, NULL,0,DIGCF_PRESENT );
	    if( INVALID_HANDLE_VALUE != hDI )
			enumDeviceName(hDI);
	    SetupDiDestroyDeviceInfoList(hDI);
	}

	printf( "\n\n" );
	return true;
}

// ---------------------------------------------
//! @brief  グラフィックスカードのスペックをテキストに出力する
//! @return trueなら成功
// ---------------------------------------------
bool outputGraphicsSpecInfo()
{
#ifdef DX9
	stl::string	str; 

	GApiDX9*	pApi = (GApiDX9*)IGraphics->getApi();

	//!	VGA
	D3DADAPTER_IDENTIFIER9	AdapterData;
	TGL::Logger::Print("\n【 VGA 】\n" );
	
	//!	チップ
	TGL::Logger::Print( "\n □ グラフィックスチップ\n" );
	pApi->getCore()->GetAdapterIdentifier( D3DADAPTER_DEFAULT, 0, &AdapterData );
	TGL::Logger::Print("  %s\n", AdapterData.Description );
	TGL::Logger::Print("  Driver : %s  version[%d.%d.%04d.%d]\n", AdapterData.Driver, HIWORD(AdapterData.DriverVersion.HighPart), LOWORD(AdapterData.DriverVersion.HighPart), HIWORD(AdapterData.DriverVersion.LowPart), LOWORD(AdapterData.DriverVersion.LowPart) );
	
	//!	詳細
	TGL::Logger::Print( "\n □ 詳細スペック \n" );
	if( ITGLDirectXDialog )
		TGL::Logger::Print("  VRAM                           : %s\n", ITGLDirectXDialog->getStrVramSize() );

	const D3DCAPS9&	d3dCaps = pApi->getCaps();
	switch( d3dCaps.VertexShaderVersion ){	//!< 頂点シェーダバージョン
		case D3DVS_VERSION(1,1):	TGL::Logger::Print( "  Vertex Shader Version          : 1.1\n" );		break;
		case D3DVS_VERSION(2,0):	TGL::Logger::Print( "  Vertex Shader Version          : 2.0\n" );		break;
		case D3DVS_VERSION(3,0):	TGL::Logger::Print( "  Vertex Shader Version          : 3.0\n" );		break;
		default:					TGL::Logger::Print( "  Vertex Shader Version          : unknown\n" );	break;
	}
	switch( d3dCaps.PixelShaderVersion ){	//!< ピクセルシェーダバージョン
		case D3DPS_VERSION(1,1):	TGL::Logger::Print( "  Pixel Shader Version           : 1.1\n" );		break;
		case D3DPS_VERSION(1,2):	TGL::Logger::Print( "  Pixel Shader Version           : 1.2\n" );		break;
		case D3DPS_VERSION(1,3):	TGL::Logger::Print( "  Pixel Shader Version           : 1.3\n" );		break;
		case D3DPS_VERSION(1,4):	TGL::Logger::Print( "  Pixel Shader Version           : 1.4\n" );		break;
		case D3DPS_VERSION(2,0):	TGL::Logger::Print( "  Pixel Shader Version           : 2.0\n" );		break;
		case D3DPS_VERSION(3,0):	TGL::Logger::Print( "  Pixel Shader Version           : 3.0\n" );		break;
		default:					TGL::Logger::Print( "  Pixel Shader Version           : unknown\n" );	break;
	}

	str = stl::nullstr();
	if( d3dCaps.FVFCaps & D3DFVFCAPS_DONOTSTRIPELEMENTS)	str += " DONOTSTRIPELEMENTS";
	if( d3dCaps.FVFCaps & D3DFVFCAPS_PSIZE)					str += " PSIZE";
	TGL::Logger::Print("  FVFCaps                        :%s\n", str.c_str() );

	TGL::Logger::Print("  MaxVShaderInstructionsExecuted : %d\n",d3dCaps.MaxVShaderInstructionsExecuted);
	TGL::Logger::Print("  MaxPShaderInstructionsExecuted : %d\n",d3dCaps.MaxPShaderInstructionsExecuted);

	TGL::Logger::Print("  MaxPrimitiveCount              : %d\n",d3dCaps.MaxPrimitiveCount);
	TGL::Logger::Print("  MaxVertexIndex                 : %d\n",d3dCaps.MaxVertexIndex);
	TGL::Logger::Print("  MaxStreams                     : %d\n",d3dCaps.MaxStreams);
	TGL::Logger::Print("  MaxStreamStride                : %d\n",d3dCaps.MaxStreamStride);
	TGL::Logger::Print("  MaxStreamStride                : %d\n",d3dCaps.MaxStreamStride);
	TGL::Logger::Print("  MaxVertexBlendMatrices         : %d\n",d3dCaps.MaxVertexBlendMatrices);
	TGL::Logger::Print("  MaxVertexBlendMatrixIndex      : %d\n",d3dCaps.MaxVertexBlendMatrixIndex);
	
	str = stl::nullstr();
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_ALPHA )				str += " ALPHA";
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_ALPHAPALETTE )		str += " ALPHAPALETTE";
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_CUBEMAP )				str += " CUBEMAP";
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_CUBEMAP_POW2 )		str += " CUBEMAP_POW2";
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_NONPOW2CONDITIONAL )	str += " NONPOW2CONDITIONAL";
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_POW2 )				str += " POW2";
	if( d3dCaps.TextureCaps & D3DPTEXTURECAPS_SQUAREONLY )			str += " SQUAREONLY";
	TGL::Logger::Print("  TextureCaps                    :%s\n", str.c_str() );

	TGL::Logger::Print("  Max Texture Size               : %d, %d\n", d3dCaps.MaxTextureWidth, d3dCaps.MaxTextureHeight );
	TGL::Logger::Print("  Max Texture Aspect Ratio       : %d\n", d3dCaps.MaxTextureAspectRatio );
	TGL::Logger::Print("  MaxTextureRepeat               : %d\n",d3dCaps.MaxTextureRepeat);
	TGL::Logger::Print("  MaxTextureAspectRatio          : %d\n",d3dCaps.MaxTextureAspectRatio);
	TGL::Logger::Print("  MaxAnisotropy                  : %d\n",d3dCaps.MaxAnisotropy);
	TGL::Logger::Print("  MaxVertexW                     : %d\n",d3dCaps.MaxVertexW);
	TGL::Logger::Print("  MaxTextureBlendStages          : %d\n",d3dCaps.MaxTextureBlendStages);
	TGL::Logger::Print("  MaxSimultaneousTextures        : %d\n",d3dCaps.MaxSimultaneousTextures);

	// -----------------------------------------
	//!	ディプレイモード
	D3DDISPLAYMODE	d3dDispMode;		//!< 現在のディスプレイモード
	D3DDISPLAYMODE	d3dDispModeEnable;	//!< 使用可能なディスプレイモード
	s32				nAdaMax;			//!< アダプタモード列挙数
	TGL::Logger::Print( "\n □ ディスプレイモード\n" );
	pApi->getCore()->GetAdapterDisplayMode( D3DADAPTER_DEFAULT, &d3dDispMode );	//!< 現在のアダプタの設定を取得
	nAdaMax = pApi->getCore()->GetAdapterModeCount( D3DADAPTER_DEFAULT, d3dDispMode.Format );
	for(int i=0 ; i<nAdaMax ; i++ ){
		pApi->getCore()->EnumAdapterModes( D3DADAPTER_DEFAULT, d3dDispMode.Format, i, &d3dDispModeEnable );
		if( d3dDispModeEnable.RefreshRate != 60 )	continue;	//!< 60Hzだけ出す

		//!	アスペクト比の計算
		stl::string	strAspect(stl::nullstr());
		f32	fAspect = (f32)d3dDispModeEnable.Height / (f32)d3dDispModeEnable.Width;
		if(      fAspect == 0.75f )			strAspect = "(4:3)  ";
		else if( fAspect == 0.8f )			strAspect = "(5:4)  ";
		else if( fAspect == 0.6f )			strAspect = "(15:9) ";
		else if( fAspect == 0.5625f )		strAspect = "(16:9) ";
		else if( fAspect == 0.625f )		strAspect = "(16:10)";
		else if( fAspect == 0.52734375f )	strAspect = "(17:9) ";
		else if( fAspect == 0.48f )			strAspect = "(25:12)";
		else if( fAspect == 0.64f )			strAspect = "(25:16)";
		else								strAspect = "(----) ";

		str = stl::FormatString<stl::string>("  ID:%2d, %4d*%4d %s, RefreshRate:%2d, Format:", i, d3dDispModeEnable.Width, d3dDispModeEnable.Height, strAspect.c_str(), d3dDispModeEnable.RefreshRate );
		switch( d3dDispModeEnable.Format ){
			case D3DFMT_A2B10G10R10:	str += "A2B10G10R10\n";	break;
			case D3DFMT_A8R8G8B8:		str += "A8R8G8B8\n";	break;
			case D3DFMT_R8G8B8:			str += "R8G8B8\n";		break;
			case D3DFMT_X8R8G8B8:		str += "X8R8G8B8\n";	break;
			case D3DFMT_A1R5G5B5:		str += "A1R5G5B5\n";	break;
			case D3DFMT_X1R5G5B5:		str += "X1R5G5B5\n";	break;
			case D3DFMT_R5G6B5:			str += "R5G6B5\n";		break;
			default:					str += "unknown\n";		break;
		}
		TGL::Logger::Print( str );
	}
#endif
	return true;
}


/*
//////////////////////////////////////////////////////////////////////
// Check CPUID / RDTSC Enabled
//////////////////////////////////////////////////////////////////////
s32 _SYS_CheckEnableCPUID()
{

#ifdef _WIN32
#ifndef _X86_64
	s32 FlagCPUID = false;
	DWORD flag_1, flag_2;
	_asm{
			pushfd
			pop		eax
			mov		flag_1, eax
			xor		eax, 00200000h
			push	eax
			popfd
			pushfd
			pop		eax
			mov		flag_2, eax
		}
    if( flag_1 == flag_2){	// Disabled CPUID
		FlagCPUID = false;
    }else{					// Enabled CPUID
		FlagCPUID = true;
	}
#else
	FlagCPUID = true;
#endif

#else
// for UNIX
	FlagCPUID = true; // (^_^;
#endif
	return FlagCPUID;
}
*/

// ---------------------------------------------
//! @brief  CPUID
//! @param  op	[]	
//! @param  EAX	[]	
//! @param  EBX	[]	
//! @param  ECX	[]	
//! @param  EDX	[]	
// ---------------------------------------------
void cpuid(DWORD op, DWORD *EAX, DWORD *EBX, DWORD *ECX, DWORD *EDX)
{
#ifdef _WIN32

#ifndef WIN64
	DWORD A, B, C, D;
	_asm{
		mov eax, op
		mov ecx, 0
		cpuid
		mov A, eax
		mov B, ebx
		mov C, ecx
		mov D, edx
	}
	*EAX = A;
	*EBX = B;
	*ECX = C;
	*EDX = D;
#else
	_cpuid(op,EAX,EBX,ECX,EDX);
#endif

#else
	__asm__("cpuid"
		:	"=a" (*EAX),
			"=b" (*EBX),
			"=c" (*ECX),
			"=d" (*EDX)
		:	"a" (op));
#endif
}

// ---------------------------------------------
//! @brief  CPU情報の出力
// ---------------------------------------------
void outputCPUInfo()
{
	char	str[256];
	char	CPUName[256]	= "";
	char	CPUType[128]	= "";
	//s32		nCpuTypeId;
	u32	uMmcType = 0;
	double	fCpuClock;
	s32		bCPUID	= false;
	s32		FlagBrand = 0;		// Intel AMD etc..
	char	ProcessorSerial[30];
	s32		MaxCPUID;
//	s32		HyperThreadNum = 0;
	// integer //
	s32			Number;
	s32			Family;
	s32			Model;
	s32			Stepping;
	s32			Type;
	s32			ModelX=0;
	s32			Apic;
	s32			FamilyEx;
	s32			ModelEx;
	s32			FamilyX;
	s32			ExFamily;	// for AMD
	s32			ExModel;	// for AMD
	s32			ExStepping;	// for AMD
	s32			ExFamilyX;	// for AMD
	s32			ExModelX;	// for AMD


	DWORD		Feature;
	DWORD		FeatureEcx;
	DWORD		FeatureEx;
	DWORD		FeatureExEcx;
//	DWORD		FeatureVia;
//	DWORD		FeatureTransmeta;
//	DWORD		FeaturePM;
	DWORD		MiscInfo;
	DWORD		MiscInfoEx;
	DWORD		Version;
	DWORD		VersionEx=0;


	SYSTEM_INFO	sysInfo;
	GetSystemInfo( &sysInfo );


	prints("\n【 CPU 】\n" );
#ifndef WIN64
	__asm
	{
		/* CPUID命令の存在チェック */
		PUSHFD
		POP		EAX
		MOV		EBX,		EAX
		XOR		EAX,		1<<21
		PUSH	EAX
		POPFD
		PUSHFD
		POP		EAX
		CMP		EAX,		EBX
		JE		L1				// ない
		MOV		bCPUID,		1

		/* CPUID 0 */
		MOV		EAX,		0
		CPUID

		CMP		EAX,		0
		JE		L1				// 0では話にならん

		MOV DWORD PTR [CPUType+0],	EBX
		MOV DWORD PTR [CPUType+8],	ECX
		MOV DWORD PTR [CPUType+4],	EDX
	L1:
	}
#endif

	if( !bCPUID )
	{
		printf("CPUID未対応\n");
	}

#ifdef CPUCACHE_DETAIL
	//!	キャッシュ定義
	// http://softwaretechnique.web.fc2.com/OS_Development/Tips/IA32_Instructions/CPUID.html
	typedef struct {
		u8		mDesc;		//!< 記述子
		char*	mpComment;	//!< 解説
	} CACHE_T;
	
	//!	L1命令キャッシュ
	const CACHE_T	_L1I_CACHE[] = {
		{ 0x00, "Unknown" },
		{ 0x06, "8KB, 4-Ways, 32-Byte lines" },
		{ 0x08, "16KB, 4-Ways, 32-Byte lines" },
		{ 0x09, "32KB, 4-Ways, 64-Byte lines" },
		{ 0x15, "16KB, 4-Ways, 32-Byte lines" },
		{ 0x77, "16KB, 4-Ways, 64-Byte lines" },
	};
	//!	L1データキャッシュ
	const CACHE_T	_L1D_CACHE[] = {
		{ 0x00, "Unknown" },
		{ 0x0A, "8KB, 2-Ways, 32-Byte lines" },
		{ 0x0C, "16KB, 4-Ways, 32-Byte lines" },
		{ 0x0D, "16KB, 4-Ways, 64-Byte lines" },
		{ 0x0E, "24KB, 6-Ways, 64-Byte lines" },
		{ 0x10, "16KB, 4-Ways, 32-Byte lines" },
		{ 0x2C, "32KB, 8-Ways, 64-Byte lines" },
		{ 0x30, "32KB, 8-Ways, 64-Byte lines" },
		{ 0x60, "16KB, 8-Ways, 64-Byte lines" },
		{ 0x66, "8KB, 4-Ways, 64-Byte lines" },
		{ 0x67, "16KB, 4-Ways, 64-Byte lines" },
		{ 0x68, "32KB, 4-Ways, 64-Byte lines" },
	};
	//!	L2キャッシュ
	const CACHE_T	_L2_CACHE[] = {
		{ 0x00, "Unknown" },
		{ 0x1A, "96KB, 6-Ways, 64-Byte lines" },
		{ 0x21, "256KB, 8-Ways, 64-Byte lines" },
		{ 0x39, "128KB, 4-Ways, 64-Byte lines" },
		{ 0x3A, "192KB, 6-Ways, 64-Byte lines" },
		{ 0x3B, "128KB, 2-Ways, 64-Byte lines" },
		{ 0x3C, "256KB, 4-Ways, 64-Byte lines" },
		{ 0x3D, "384KB, 6-Ways, 64-Byte lines" },
		{ 0x3E, "512KB, 4-Ways, 64-Byte lines" },
		{ 0x41, "128KB, 4-Ways, 32-Byte lines" },
		{ 0x42, "256KB, 4-Ways, 32-Byte lines" },
		{ 0x43, "512KB, 4-Ways, 32-Byte lines" },
		{ 0x44, "1MB, 4-Ways, 32-Byte lines" },
		{ 0x45, "2MB, 4-Ways, 32-Byte lines" },
		{ 0x48, "3MB, 12-Ways, 64-Byte lines" },
		{ 0x49, "4MB, 16-Ways, 64-Byte lines" },
		{ 0x4E, "6144KB, 24-Ways, 64-Byte lines" },
		{ 0x78, "1MB, 8-Ways, 64-Byte lines" },
		{ 0x79, "128KB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x7A, "256KB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x7B, "512KB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x7C, "1MB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x7D, "2MB, 8-Ways, 64-Byte lines" },
		{ 0x7E, "256KB, 8-Ways, 128-Byte lines" },
		{ 0x7F, "512KB, 2-Ways, 64-Byte lines" },
		{ 0x80, "512KB, 8-Ways, 64-Byte lines" },
		{ 0x81, "128KB, 8-Ways, 32-Byte lines" },
		{ 0x82, "256KB, 8-Ways, 32-Byte lines" },
		{ 0x83, "512KB, 8-Ways, 32-Byte lines" },
		{ 0x84, "1MB, 8-Ways, 32-Byte lines" },
		{ 0x85, "2MB, 8-Ways, 32-Byte lines" },
		{ 0x86, "512KB, 4-Ways, 64-Byte lines" },
		{ 0x87, "1MB, 8-Ways, 64-Byte lines" },
	};
	//!	L3キャッシュ
	const CACHE_T	_L3_CACHE[] = {
		{ 0x00, "Unknown" },
		{ 0x22, "512KB, 4-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x23, "1MB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x25, "2MB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x29, "4MB, 8-Ways, 64-Byte lines, 128-Byte sectors" },
		{ 0x46, "4MB, 4-Ways, 64-Byte lines" },
		{ 0x47, "8MB, 8-Ways, 64-Byte lines" },
		{ 0x49, "4MB, 16-Ways, 64-Byte lines" },
		{ 0x4A, "6MB, 12-Ways, 64-Byte lines" },
		{ 0x4B, "8MB, 16-Ways, 64-Byte lines" },
		{ 0x4C, "12MB, 12-Ways, 64-Byte lines" },
		{ 0x4D, "16MB, 16-Ways, 64-Byte lines" },
		{ 0x88, "2MB, 4-Ways, 64-Byte lines" },
		{ 0x89, "4MB, 4-Ways, 64-Byte lines" },
		{ 0x8A, "8MB, 4-Ways, 64-Byte lines" },
		{ 0x8D, "3MB, 12-Ways, 128-Byte lines" },
		{ 0xD0, "512KB page, 4-Ways, 64-Byte lines" },
		{ 0xD1, "1MB page, 4-Ways, 64-Byte lines" },
		{ 0xD2, "2MB page, 4-Ways, 64-Byte lines" },
		{ 0xD6, "1MB page, 8-Ways, 64-Byte lines" },
		{ 0xD7, "2MB page, 8-Ways, 64-Byte lines" },
		{ 0xD8, "4MB page, 8-Ways, 64-Byte lines" },
		{ 0xDC, "1.5MB page, 12-Ways, 64-Byte lines" },
		{ 0xDD, "3MB page, 12-Ways, 64-Byte lines" },
		{ 0xDE, "6MB page, 12-Ways, 64-Byte lines" },
		{ 0xE2, "2MB page, 16-Ways, 64-Byte lines" },
		{ 0xE4, "8MB page, 16-Ways, 64-Byte lines" },
		{ 0xEA, "12MB page, 24-Ways, 64-Byte lines" },
		{ 0xEB, "18MB page, 24-Ways, 64-Byte lines" },
		{ 0xEC, "24MB page, 24-Ways, 64-Byte lines" },
	};
	const CACHE_T*	pL1I = &_L1I_CACHE[0];
	const CACHE_T*	pL1D = &_L1D_CACHE[0];
	const CACHE_T*	pL2  = &_L2_CACHE[0];
	const CACHE_T*	pL3  = &_L3_CACHE[0];
#else	/*	CPUCACHE_DETAIL	*/
	u32		uL2 = 0;
	u32		uL3 = 0;
#endif	/*	CPUCACHE_DETAIL	*/

	//long TypeID = 0; // CPUType
	char bTSC;

	u16 mCpuCore;
	
	if( bCPUID )
	{


		//===============================
		// Family/Model/Stepping
		//===============================
		{

			DWORD EAX, EBX, ECX, EDX;

			Number = sysInfo.dwNumberOfProcessors;

			// Family / Model / Step
			cpuid(0x1, &EAX, &EBX, &ECX, &EDX);

			Version		= EAX;
			MiscInfo	= EBX;
			Feature		= EDX;
			FeatureEcx	= ECX;
			Family		= (Version>>8) & 0xF;
			Model		= (Version>>4) & 0xF;
			Stepping	= Version & 0xF;
			Type		= (Version>>12) & 0x3;
			if(Family == 0xF || (Family == 6 && ModelX > 0xE)){
				Apic	= (MiscInfo>>24) & 0xFF;
			}

			char		TypeName[33];
			switch( Type )
			{
			case 0:	strcpy(TypeName,"Original OEM processor");	break;
			case 1:	strcpy(TypeName,"OverDrive processor");		break;
			case 2:	strcpy(TypeName,"Dual processor");			break;
			default:strcpy(TypeName,"Reserved");				break;
			}

			FamilyEx = (Version>>20) & 0xFF;
			ModelEx = (Version>>16) & 0xF;
			FamilyX = FamilyEx + Family;
			ModelX = ModelEx * 16 + Model;

			if(FlagBrand == INTEL)
			{
				ExFamilyX = FamilyX;
				ExModelX = ModelX;
				ExStepping = Stepping;
			}

			if(FlagBrand != INTEL || Family == 0xF){
				FamilyEx = (VersionEx>>20) & 0xFF;
				ModelEx = (VersionEx>>16) & 0xF;
				ExFamily   = (VersionEx>>8) & 0xF;
				ExFamilyX = FamilyEx + ExFamily;
				ExModel    = (VersionEx>>4) & 0xF;
				ExModelX = ModelEx * 16 + ExModel;
				ExStepping = VersionEx & 0xF;
			}

		}

		
		//===============================
		// ベンダー名
		//===============================
		{
			char		VendorString[13];			// OriginalVendorString
			char		VendorName[13];				// VendorName
			

			DWORD EAX, EBX, ECX, EDX;

			// VendorName
			char vendor[13];
			vendor[12] = '\0';
			cpuid(0x0, &EAX, &EBX, &ECX, &EDX);

			MaxCPUID = EAX;
			memcpy(vendor  , &EBX, 4);
			memcpy(vendor+4, &EDX, 4);
			memcpy(vendor+8, &ECX, 4);
			sprintf(VendorString,vendor);

			if( strcmp(VendorString,"GenuineIntel") == 0 ){
				FlagBrand = INTEL;
				sprintf(VendorName,"Intel");
			}else if( strcmp(VendorString,"AuthenticAMD") == 0 ){
				FlagBrand = AMD;
				sprintf(VendorName,"AMD");
			}else if( strcmp(VendorString,"GenuineTMx86") == 0 ){
				FlagBrand = TMx86;
				sprintf(VendorName,"Transmeta");
			}else if( strcmp(VendorString,"CyrixInstead") == 0 ){
				FlagBrand = CYRIX;
				sprintf(VendorName,"Cyrix");
			}else if( strcmp(VendorString,"CentaurHauls") == 0 ){
				FlagBrand = IDT;
				sprintf(VendorName,"IDT");
			}else if( strcmp(VendorString,"RiseRiseRise") == 0 ){
				FlagBrand = RISE;
				sprintf(VendorName,"Rise");
			}else if( strcmp(VendorString,"NexGenDriven") == 0 ){
				FlagBrand = NEXGEN;
				sprintf(VendorName,"NexGen");
			}else if( strcmp(VendorString,"UMC UMC UMC ") == 0 ){
				FlagBrand = UMC;
				sprintf(VendorName,"UMC");
			}else if( strcmp(VendorString,"Geode By NSC") == 0 ){
				FlagBrand = NSC;
				sprintf(VendorName,"NSC");
			}else if( strcmp(VendorString,"Compaq FX32!") == 0 ){
				FlagBrand = COMPAQ;
				sprintf(VendorName,"Compaq");
			}

			printf("  ベンダー           : %s\n",VendorName);
		}
		
		//===============================
		// CPUの名前
		//===============================
		{

			DWORD EAX, EBX, ECX, EDX;
			char name[49];
			name[48]='\0';
			s32 i=0;
			for( s32 j = 2; j <= 4; j++){
				cpuid(0x80000000 + j, &EAX, &EBX, &ECX, &EDX);
				memcpy(name + 4 * (i++), &EAX, 4);
				memcpy(name + 4 * (i++) ,&EBX, 4);
				memcpy(name + 4 * (i++), &ECX, 4);
				memcpy(name + 4 * (i++), &EDX, 4);
			}
			char *p;
			p = name;
			while(*p == ' '){p++;}

			sprintf(CPUName,p);

			//	CPU名
			if( strlen(CPUName) <= 0 )
			{
				switch( FlagBrand ){
					case INTEL:

						switch( Family ){
						case 0xF:
							strcpy( CPUName, "Pentium 4系" );
							break;
						case 0x7:
							strcpy( CPUName, "Itanium (Merced)" );
							break;
						case 0x6:
							strcpy( CPUName, "Core系(Core/Core2/Xeon/Core i7)" );
							break;
						default:
							strcpy( CPUName, "UnKnown Intel CPU" );	// 取得できなかったとき
							break;
						}
						break;
					case AMD:

						strcpy( CPUName, "UnKnown AMD CPU" );	// 取得できなかったとき

						break;
					default:
						strcpy( CPUName, "UnKnown" );	// 取得できなかったとき
						break;
				
				}
			}
			printf("  CPU名称            : %s\n", CPUName );

		}





		//===============================
		// 命令
		//===============================
		{
			DWORD MaxCPUIDEx;

			// Extended CPUID
			DWORD EAX, EBX, ECX, EDX;
			cpuid(0x80000000, &EAX, &EBX, &ECX, &EDX);
			MaxCPUIDEx = EAX;// FeatureEx

			if( MaxCPUIDEx > 0x80000000 ){
				cpuid(0x80000001, &EAX, &EBX, &ECX, &EDX);
				VersionEx = EAX;
				MiscInfoEx = EBX;
				FeatureEx = EDX;
				FeatureExEcx = ECX;
			}
			
			//!	キャッシュ
#ifdef CPUCACHE_DETAIL
			cpuid(0x00000002, &EAX, &EBX, &ECX, &EDX);
			if( EAX & 0x01 ){
				const u32	_DESC[] = {	//!< 比較記述子
					(EAX & 0x000000FF) >> 0,
					(EAX & 0x0000FF00) >> 8,
					(EAX & 0x00FF0000) >> 16,
					(EAX & 0xFF000000) >> 24,
					(EBX & 0x000000FF) >> 0,
					(EBX & 0x0000FF00) >> 8,
					(EBX & 0x00FF0000) >> 16,
					(EBX & 0xFF000000) >> 24,
					(ECX & 0x000000FF) >> 0,
					(ECX & 0x0000FF00) >> 8,
					(ECX & 0x00FF0000) >> 16,
					(ECX & 0xFF000000) >> 24,
					(EDX & 0x000000FF) >> 0,
					(EDX & 0x0000FF00) >> 8,
					(EDX & 0x00FF0000) >> 16,
					(EDX & 0xFF000000) >> 24,
				};
				for( u8 i=0 ; i<numberof(_DESC) ; i++ ){
					if( !_DESC[i] )	continue;
					//!	L1命令キャッシュ
					for( u8 j=1 ; j<numberof(_L1I_CACHE) ; j++ ){
						if( _DESC[i] != _L1I_CACHE[j].mDesc )	continue;
						pL1I = &_L1I_CACHE[j];
						break;
					}
					//!	L1データキャッシュ
					for( u8 j=1 ; j<numberof(_L1D_CACHE) ; j++ ){
						if( _DESC[i] != _L1D_CACHE[j].mDesc )	continue;
						pL1D = &_L1D_CACHE[j];
						break;
					}
					//!	L2キャッシュ
					for( u8 j=1 ; j<numberof(_L2_CACHE) ; j++ ){
						if( _DESC[i] != _L2_CACHE[j].mDesc )	continue;
						pL2 = &_L2_CACHE[j];
						break;
					}
					//!	L3キャッシュ
					for( u8 j=1 ; j<numberof(_L3_CACHE) ; j++ ){
						if( _DESC[i] != _L3_CACHE[j].mDesc )	continue;
						pL3 = &_L3_CACHE[j];
						break;
					}
				}
			}
#else	/*	CPUCACHE_DETAIL	*/
			if( MaxCPUIDEx > 0x80000006 )
			{
				//EAX=0x80000006(cache information)
				cpuid(0x80000006, &EAX, &EBX, &ECX, &EDX);
				uL2 = (ECX) >> 16;
				uL3 = (EDX) >> 16;
			}
#endif	/*	CPUCACHE_DETAIL	*/

			if( (FeatureEx >> 22) & 0x1 ) uMmcType|=CPUF_MMXEx;

			
			if( FlagBrand == INTEL )
			{
				if( (FeatureEx >> 29) & 0x1 ) uMmcType|=CPUF_IA32e;
			}else{
				if( (FeatureEx >> 29) & 0x1 ) uMmcType|=CPUF_AA64;
			}

			if( (FeatureEx >> 31) & 0x1 )	uMmcType|=CPUF_3DNow;
			if( (FeatureEx >> 30) & 0x1 )	uMmcType|=CPUF_3DNowEx;
			if( (FeatureEx >> 20) & 0x1 )	uMmcType|=CPUF_NX;
			if( (FeatureExEcx >> 2) & 0x1 ) uMmcType|=CPUF_AmdV;
			if( (FeatureExEcx >> 6) & 0x1 ) uMmcType|=CPUF_SSE4A;
			if( (FeatureExEcx >> 11) & 0x1) uMmcType|=CPUF_SSE5;
			if( (FeatureEcx >> 28) & 0x1  ) uMmcType|=CPUF_AVX;

			if( (Feature>> 5) & 0x1 )	uMmcType|=CPUF_MSR;
			if( (Feature>>23) & 0x1)	uMmcType|=CPUF_MMX;
			if( (Feature>>25) & 0x1 )	uMmcType|=CPUF_SSE;
			if( (Feature>>26) & 0x1 )	uMmcType|=CPUF_SSE2;
			if( (FeatureEcx ) & 0x1 )	uMmcType|=CPUF_SSE3;
			if( (FeatureEcx>>9) & 0x1 )	uMmcType|=CPUF_SSSE3;
			if( (FeatureEcx >> 19) & 0x1 )	uMmcType|=CPUF_SSE41;
			if( (FeatureEcx >> 20) & 0x1 )	uMmcType|=CPUF_SSE42;
			if( (FeatureEcx>>5) & 0x1)	uMmcType|=CPUF_VT;
			if( (Feature>>28) & 0x1	)	uMmcType|=CPUF_HT;
			if( (Feature>>30) & 0x1 )	uMmcType|=CPUF_IA64;

			if( (uMmcType & CPUF_SSE41) || (uMmcType & CPUF_SSE42) )
			{
				uMmcType|=CPUF_SSE4;
			}


			ProcessorSerial[0] = '\0';
			// Processor Serial Number
			if( MaxCPUID >= 3 && ((Feature>>18) & 0x1) )
			{
				cpuid(0x1, &EAX, &EBX, &ECX, &EDX);
				cpuid(0x3, &EBX, &EBX, &ECX, &EDX); // EBX is Dummy
				sprintf(ProcessorSerial,
					"%04X-%04X-%04X-%04X-%04X-%04X",
					(EAX >> 16), (EAX & 0xFFFF),
					(EDX >> 16), (EDX & 0xFFFF),
					(ECX >> 16), (ECX & 0xFFFF)
					);
			}

			//!		CPUコア		
#if 1
			SYSTEM_INFO	si;
			GetSystemInfo( &si );
			mCpuCore = (u16)si.dwNumberOfProcessors;
#else
			//!	HyperThread
			cpuid(0x1, &EAX, &EBX, &ECX, &EDX);
			
			if( (uMmcType & CPUF_HT) && FlagBrand == INTEL && Family == 0xF){ // EnableHyperThreading
				HyperThreadNum = (EBX>>16) & 0xFF;
			}else if((uMmcType & CPUF_HT) && FlagBrand == INTEL){ // Conroe , Merom
				mCpuCore = (EBX>>16) & 0xFF;
				HyperThreadNum = 0;
			}else{
				HyperThreadNum = 0;
			}

			//!	コア数
			//!	AMD
			if(FlagBrand == AMD && MaxCPUIDEx >= 0x80000008){
				cpuid(0x80000008, &EAX, &EBX, &ECX, &EDX);
				mCpuCore = (ECX & 0xFF) + 1;
			}

			//!	Intel
			if(FlagBrand == INTEL && MaxCPUID >= 0x00000004){
				cpuid(0x00000004, &EAX, &EBX, &ECX, &EDX);
				mCpuCore = ((EAX >> 26) & 0x3F) + 1;
			}
			//! for Pentium D, Pentium XE
			if(mCpuCore >= 2 && FlagBrand == INTEL && Family == 0xF){
				HyperThreadNum = HyperThreadNum / mCpuCore;
			}
#endif

			// CPUクロック計測
			if( (Feature>> 4) & 0x1 ) bTSC = true;
		}

		{
			//=======================
			// アーキテクチャ
			//=======================
			char Architecture[24];
			switch(sysInfo.wProcessorArchitecture)
			{
			case PROCESSOR_ARCHITECTURE_INTEL:
				strcpy(Architecture,"x86");
				break;
			case PROCESSOR_ARCHITECTURE_MIPS:
				strcpy(Architecture,"MIPS");
				break;
			case PROCESSOR_ARCHITECTURE_ALPHA:
				strcpy(Architecture,"Alpha");
				break;
			case PROCESSOR_ARCHITECTURE_PPC:
				strcpy(Architecture,"PowerPC");
				break;
			case PROCESSOR_ARCHITECTURE_IA64:
				strcpy(Architecture,"IA64");
				break;
			case PROCESSOR_ARCHITECTURE_AMD64:
				strcpy(Architecture,"x64");
				break;
			case PROCESSOR_ARCHITECTURE_IA32_ON_WIN64:
				strcpy(Architecture,"WOW64");
				break;
			case PROCESSOR_ARCHITECTURE_SHX:
			case PROCESSOR_ARCHITECTURE_ARM:
			case PROCESSOR_ARCHITECTURE_ALPHA64:
			case PROCESSOR_ARCHITECTURE_MSIL:
				strcpy(Architecture,"not supported");
				break;
			case PROCESSOR_ARCHITECTURE_UNKNOWN:
			default:
				// 未知のアーキテクチャ
				if( IsWow64() ){
					sprintf(Architecture,"unknown x64");
				}else
				{
					sprintf(Architecture,"unknown x86");
				}
				break;
			}
			printf("  アーキテクチャ     : %s [Family:%d Model:%d Stepping:%d]\n", Architecture,sysInfo.wProcessorLevel,Model,Stepping );
			//=============================
		}	
	}


	/*=============[ CPUクロック取得 ]================*/
	double CPUClock		= 0.0;	
	if ( bTSC ){
#ifndef WIN64
		LARGE_INTEGER cycle;
		__asm {
            cpuid
            rdtsc
            mov   cycle.LowPart, eax
            mov   cycle.HighPart, edx
        }

        Sleep(1000);

        __asm {
            cpuid
            rdtsc
            sub   eax, cycle.LowPart
            sub   edx, cycle.HighPart
            mov   cycle.LowPart, eax
            mov   cycle.HighPart, edx
        }
        CPUClock = (double)cycle.LowPart;
        CPUClock /= 1000000;
#endif
	}
	fCpuClock = CPUClock;


	
	//sprintf( str, "  Serial   : %s\n", ProcessorSerial );	TGL::Logger::Print( str );
	//sprintf( str, "  CoreNum  : %d\n", sysInfo.dwNumberOfProcessors );	TGL::Logger::Print( str );
	sprintf( str, "  コア数             : %d\n", mCpuCore );		TGL::Logger::Print( str );
//	sprintf( str, "  ハイパースレッド   : %d\n", HyperThreadNum );	TGL::Logger::Print( str );

	sprintf( str, "  クロック数         : %0.2f[MHz]\n", fCpuClock );	TGL::Logger::Print( str );

	// 機能
	const char func[][32] = {	"MMX","MMXEx","SSE","SSE2","SSE3","SSSE3","SSE4","SSE41","SSE42","SSE4A","SSE5",
					"AVX","3DNow","3DNowEx","HT","VT","AmdV","AA64","IA64",
					"SpeedStep","EIST","PowerNow","LongHaul","LongRun","ClockModulation","ProcessorSerial",
					"Intel64","NX","MSR","K7Sempron","DualCore","EistCorrect" };

	QString strFunc;
	for(s32 i = 0;i<32;i++)
	{
		if( tstFLG(uMmcType,1<<i ) )
		{
			strFunc += func[i];
			strFunc += " ";
		}
	}

	printf("  対応機能           : %s\n",strFunc.toStdString());

	
#ifdef CPUCACHE_DETAIL
	TGL::Logger::Print("  L1命令キャッシュ   : %s\n", pL1I->mpComment );
	TGL::Logger::Print("  L1データキャッシュ : %s\n", pL1D->mpComment );
	TGL::Logger::Print("  L2キャッシュ       : %s\n", pL2->mpComment );
	TGL::Logger::Print("  L3キャッシュ       : %s\n", pL3->mpComment );
#else	/*	CPUCACHE_DETAIL	*/
	if( uL2 )	printf("  L2キャッシュ       : %d KB\n", uL2 );
	if( uL3 )	printf("  L3キャッシュ       : %d KB\n", uL3 );
#endif	/*	CPUCACHE_DETAIL	*/
}
/* ========================================================================= */
/*		EOF																	 */
/* ========================================================================= */
#endif