/** \class TGL::Logger
 * 
 * @brief	ログ出力
 *
 * @author	DM
 *
 * @date	2008/11/09 新規作成
 *****************************************************************************/


#pragma	once

#include "cTDate.h"
#include <string>

#define PRINT(format,...)		do { TGL::Logger::Print(format,__VA_ARGS__);} while (false)
#define T_LOG(format,...)		do { TGL::Logger::Log(format,__VA_ARGS__);TGL::Logger::OutputSourcePath(__FILE__, __LINE__);  } while (false)
#define T_WARNING(format,...)	do { TGL::Logger::Warning(format,__VA_ARGS__);TGL::Logger::OutputSourcePath(__FILE__, __LINE__); } while (false)
#define T_ERROR(format,...)		do { TGL::Logger::Error(format,__VA_ARGS__);TGL::Logger::OutputSourcePath(__FILE__, __LINE__);ASSERTMSG(0,format,__VA_ARGS__);} while (false)

namespace TGL
{
	using namespace std;

//! ログ出力
class Logger
{
public:

	//! 初期化
	//! @memo 初期化の前にSetCurrentDirectoryで適切なフォルダを指定してください。
	static void Initialize(const char* pszTitleName,const char* pszClassName);

	//! 終了
	static void Terminate();

	// 定期更新
	//static void Update();

	//! 描画
	static void Draw();

	//! ログファイル名
	static const char* GetLogFileName();

	//! 起動時間
	static const Date& GetStartTime();

public:

	// ログ出力
	static void Print(const char * fmt, ...);
	static void Print( string& str )	{ Print( str.c_str() ); }

	// 時間つきログ出力
	static void Log(const char * fmt, ...);
	// 時間つきログ出力(警告)
	static void Warning(const char * fmt, ...);
	// 時間つきログ出力(エラー)
	static void Error(const char * fmt, ...);

	// ソースパス出力
	static void OutputSourcePath(const char *file, int line);
private:

	static void printCore(const char * fmt, ...);

	static FILE* s_fp;

	// ログファイル名
	static string s_LogFileName;

	// 起動時間
	static Date s_startTime;
};

}	// namespace TGL


//=============================================================================
// EOF
//=============================================================================
