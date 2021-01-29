/**
 * @file	TGLLogger.cpp
 * @brief	ログ出力
 *
 * @date	2008/11/09
 * @author	DM
  *****************************************************************************/
/* ====< インクルード >====================================================== */

// 排他制御
//#include "../Core/Thread/TGLCriticalSection.h"

#include "TLogger.h"
#include <stdio.h>
#include <string>
#include <stdarg.h>
#include <debugapi.h>

#define ENABLE_INSERT_MODE 0x0020
#define ENABLE_QUICK_EDIT_MODE 0x0040
#define ENABLE_EXTENDED_FLAGS 0x0080
#define ENABLE_AUTO_POSITION 0x0100

namespace TGL
{
	using namespace std;

	typedef signed int s32;
	typedef int u32;

#define T_CONSOLE(str)			do { OutputDebugString( str );} while (false)


// local namespace
namespace
{
	// ログファイルのファイルポインタ
	const char* LOG_FILE_EXT = ".log";
	FILE *fp = 0;
	//HANDLE hIn, hOut, hOrig;

	const s32 STR_BUFFER_SIZE = 512;

	static bool s_Init = false;

	static s32 s_AssertNum = 0;

	static u32 s_LogCount = 0;

	template<typename string>
	inline string FormatString(const char* format, ...)
	{
		const u32	MAX_BUFFER = 1024;
		char textString[MAX_BUFFER] = { '\0' };

		// -- empty the buffer properly to ensure no leaks.
		memset(textString, '\0', sizeof(textString));

		va_list	args;
		va_start(args, format);
		vsnprintf(textString, MAX_BUFFER * 5, format, args);
		va_end(args);
		return string(textString);
	}

	// 時間を文字列で取得
	string getTimeString()
	{
		Date	stTime;
		stTime.getNow();
		string str = FormatString<string>("%02d:%02d:%02d" ,
					//stTime.mYear , stTime.mMonth , stTime.mDay ,
					stTime.mHour , stTime.mMinute , stTime.mSecond
				);

		return str;
	}

};
 
FILE* Logger::s_fp = 0;
string Logger::s_LogFileName = "";
Date Logger::s_startTime;

//=============================================================================
// class Logger
//=============================================================================

//=============================================================================
//! 初期化
//-----------------------------------------------------------------------------
//=============================================================================
void Logger::Initialize(const char* pszTitleName,const char* pszClassName)
{

	// ログファイル名
	s_LogFileName = pszClassName;
	s_LogFileName += LOG_FILE_EXT;

	if (!(s_fp = fopen(s_LogFileName.c_str(), "w")))
	{
		//MsgBox(IMain->getHWnd(), "ログファイルが作成できませんでした");
		return;
	}

	setvbuf(stdout, (char *)NULL, _IONBF, 0);

	/*::AllocConsole();

	
	SetConsoleTitle(pszConsoleWindowTitle);
	HWND hwndFound = FindWindow(NULL, pszConsoleWindowTitle);

	DWORD mode;
	GetConsoleMode( hwndFound, &mode);
	SetConsoleMode( hwndFound, mode  | ENABLE_PROCESSED_INPUT);
	//SetWindowPos(hwndFound,HWND_TOP,100,100,100,10,SWP_SHOWWINDOW|SWP_NOACTIVATE );


	SetConsoleCtrlHandler((PHANDLER_ROUTINE)HandlerRoutine, true);

	_tfreopen_s(&fp, "CONOUT$", "w", stdout); // 標準出力の割り当て
	*/

	s_Init = true;
	s_LogCount = 0;

	//=================
	// ヘッダ
	//=================

	// 起動日時の取得
	s_startTime.getNow();

	string str("=========================================================\n");
	str += FormatString<string>("\t%s\nCompiled : %s %s\n",pszTitleName,__DATE__, __TIME__ );
	str += FormatString<string>("Execute : %4d/%02d/%02d %02d:%02d:%02d\n",s_startTime.mYear,s_startTime.mMonth,s_startTime.mDay,s_startTime.mHour,s_startTime.mMinute,s_startTime.mSecond);
	str += string("=========================================================\n");
	Print(str.c_str());
}

//=============================================================================
//! 終了
//-----------------------------------------------------------------------------
//=============================================================================
void Logger::Terminate()
{
	s_Init = false;

	if(s_fp) fclose(s_fp);
}

//=============================================================================
//! ログファイル名
//-----------------------------------------------------------------------------
//=============================================================================
const char* Logger::GetLogFileName()
{
	return s_LogFileName.c_str();
}

//=============================================================================
//! 起動時間
//=============================================================================
const Date& Logger::GetStartTime()
{
	return s_startTime;
}

//=============================================================================
//! ログ表示
//-----------------------------------------------------------------------------
//=============================================================================
void Logger::Draw()
{
}


//=============================================================================
// 時間つきログ出力
//=============================================================================
void Logger::Log( const char * fmt, ... )
{
	//if( !s_Init )	return;

	char	buf[STR_BUFFER_SIZE];
	va_list	ap;
	va_start( ap, fmt );
	vsnprintf( buf,STR_BUFFER_SIZE-1, fmt, ap );
	va_end( ap );

	printCore("[Log #%03d](%s) %s\n",s_LogCount,getTimeString().c_str(),buf);


	s_LogCount++;
}

//=============================================================================
// 時間つきログ出力(警告)
//=============================================================================
void Logger::Warning( const char * fmt, ... )
{
	//if( !s_Init )	return;

	char	buf[STR_BUFFER_SIZE];
	va_list	ap;
	va_start( ap, fmt );
	vsnprintf( buf,STR_BUFFER_SIZE-1, fmt, ap );
	va_end( ap );

	// 背景色設定
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(hOut, &csbi);
	SetConsoleTextAttribute(hOut, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_INTENSITY);

	printCore("[Warning #%03d](%s) %s\n",s_LogCount,getTimeString().c_str(),buf);
	
	SetConsoleTextAttribute(hOut, csbi.wAttributes);

	s_LogCount++;
}

//=============================================================================
// 時間つきログ出力(エラー)
//=============================================================================
void Logger::Error(const char * fmt, ...)
{
	if( !s_Init )	return;

	if( ++s_AssertNum > 3 ){
		return;
	}

	char	buf[STR_BUFFER_SIZE];
	va_list	ap;
	va_start( ap, fmt );
	vsnprintf( buf,STR_BUFFER_SIZE-1, fmt, ap );
	va_end( ap );

#if defined(_WINDOWS) && defined(_TGL_RELEASE)
	// 背景色設定
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(hOut, &csbi);
	SetConsoleTextAttribute(hOut, BACKGROUND_RED|0xF);
#endif

	printCore("[Error- #%03d](%s) %s\n", s_LogCount, getTimeString().c_str(), buf);

#if defined(_WINDOWS) && defined(_TGL_RELEASE)
	SetConsoleTextAttribute(hOut, csbi.wAttributes);
	
	// ウィンドウを手前に
	SetConsoleTitle(pszConsoleWindowTitle);
	HWND hwndFound = FindWindow(NULL, pszConsoleWindowTitle);
	SetForegroundWindow(hwndFound);
#endif

	s_LogCount++;
}

//=============================================================================
//! 通常のログ出力
//=============================================================================
void Logger::Print(const char * fmt, ...)
{
	if( !s_Init )	return;

	char	buf[STR_BUFFER_SIZE];
	va_list	ap;
	va_start( ap, fmt );
	vsnprintf( buf,STR_BUFFER_SIZE-1, fmt, ap );
	va_end( ap );

	printCore(buf);
}


void Logger::OutputSourcePath(const char *file, s32 line)
{
#ifndef _TGL_FINAL
	printCore("%s(%u)\n",file, line);
#endif
}

void Logger::printCore(const char * fmt, ...)
{

	//ASSERT(s_fp&&"ログ出力の準備ができてない");

#if 1
	//! ロック

	// 
	char	buf[STR_BUFFER_SIZE];
	va_list	ap;
	va_start( ap, fmt );
	vsnprintf( buf,STR_BUFFER_SIZE-1, fmt, ap );
	va_end( ap );

#if !defined(_TGL_FINAL)
	// 統合環境向け
	T_CONSOLE( buf );
//#ifdef _TGL_RELEASE

#if 1
	if( string(buf).find("====") != std::string::npos ||
		string(buf).find("Log")   != std::string::npos ||
		string(buf).find("[")   != std::string::npos )
	{
		// 区切り
		// 背景色設定
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
		GetConsoleScreenBufferInfo(hOut, &csbi);
		SetConsoleTextAttribute(hOut, FOREGROUND_GREEN | FOREGROUND_INTENSITY);
	
		printf_s(buf);
	
		SetConsoleTextAttribute(hOut, csbi.wAttributes);
	}else
#endif
	{
		// コンソールに出力
		printf_s(buf);
	}

//#endif	/*	_TGL_RELEASE	*/
#endif	/*	!defined(_TGL_FINAL)	*/


#ifdef _TGL_FINAL	//!< FINALの場合、開きっぱなしにしてると落ちる…
	if ( !(s_fp=fopen(s_LogFileName.c_str(),"at")) )
	{
		MsgBox( IMain->getHWnd(), "ログファイルが開けません。");
	}else
	{
		vfprintf(s_fp,buf, (char *)(&buf+1) );
		fclose(s_fp);
	}
#else
	//vfprintf(s_fp,buf, (char *)(&buf+1) );
#endif	/*	_TGL_FINAL	*/

#endif	/*	_ENABLE_LOG	*/
}


} // namespace TGL


//=============================================================================
// EOF
//=============================================================================
