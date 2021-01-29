// -------------------------------------------------------------------------- //
//! @file   cTGLDate.h
//! @brief  日時刻
//! @author Nal
//! @since  14/08/28(木)
//!
//! COPYRIGHT (C) 2014 EDELWEISS ALL RIGHT RESERVED
// -------------------------------------------------------------------------- //
#pragma	once
/* ====< インクルード >====================================================== */
#include	<time.h>
#include	<Windows.h>

#ifdef PS4
#include	<rtc.h>
#endif

/* ========================================================================= */
//		Date																 */
/* ========================================================================= */
namespace TGL
{
	typedef unsigned	short	u16;

class Date
{
public:
	Date()	{ clear(); }
	~Date() {}

#ifdef PS4
	void	operator = ( const SceRtcTick& time ) {
		SceRtcDateTime	date;
		sceRtcSetTick( &date, &time );
		mYear		= date.year;
		mMonth		= date.month;
		mDay		= date.day;
		mHour		= date.hour;
		mMinute		= date.minute;
		mSecond		= date.second;
		mMillisec	= date.microsecond / 1000;
	}
#else	/*	_WINDOWS	*/
	void	operator = ( const SYSTEMTIME& time ) {
		mYear		= time.wYear;
		mMonth		= time.wMonth;
		mDayOfWeek	= time.wDayOfWeek;
		mDay		= time.wDay;
		mHour		= time.wHour;
		mMinute		= time.wMinute;
		mSecond		= time.wSecond;
		mMillisec	= time.wMilliseconds;
	}
#endif	/*	_WINDOWS	*/

	bool	operator > ( const Date& date ) const {
		return difftime( getCTime(), date.getCTime() ) > 0.0;
	}

	bool	operator >= ( const Date& date ) const {
		return difftime( getCTime(), date.getCTime() ) >= 0.0;
	}

	bool	operator < ( const Date& date ) const {
		return difftime( getCTime(), date.getCTime() ) < 0.0;
	}

	bool	operator <= ( const Date& date ) const {
		return difftime( getCTime(), date.getCTime() ) <= 0.0;
	}

public:	
	// -----------------------------------------------------------------
	//		定義		
	// -----------------------------------------------------------------

	// -----------------------------------------------------------------
	//		関数		
	// -----------------------------------------------------------------
	//!	クリア
	void	clear() {
		mYear		= 0;
		mMonth		= 0;
		mDayOfWeek	= 0;
		mDay		= 0;
		mHour		= 0;
		mMinute		= 0;
		mSecond		= 0;
		mMillisec	= 0;
	}
	
	//!	現在時刻の取得
	void	getNow() {
	#ifdef PS4
		SceRtcTick	now;
		sceRtcGetCurrentTick( &now );
		*this = now;
	#else	/*	_WINDOWS	*/
		SYSTEMTIME	now;
		GetSystemTime( &now );
		*this = now;
	#endif	/*	_WINDOWS	*/
	}

	//!	C標準時刻取得
	time_t	getCTime() const {
		Date	time( *this );
		if( time.mYear < 1900 ){	//!< 不正日時
			time.mYear	= 1970;
			time.mMonth	= 1;
			time.mDay	= 1;
		}

		struct tm	t;
		ZeroMemory( &t, sizeof(t) );
		t.tm_sec  = time.mSecond;
	    t.tm_min  = time.mMinute;
	    t.tm_hour = time.mHour;
	    t.tm_mday = time.mDay;
	    t.tm_mon  = time.mMonth -1;
	    t.tm_year = time.mYear - 1900;
	    t.tm_wday = time.mDayOfWeek;

		return mktime(&t);
	}

	//!	同日判定
	bool	isSameDay( const Date& date ) const		{ return mYear+mMonth+mDay == date.mYear + date.mMonth + date.mDay; }
	//!	同時判定
	bool	isSameHour( const Date& date ) const	{ return mYear+mMonth+mDay+mHour == date.mYear + date.mMonth + date.mDay + date.mHour; }
	//!	同分判定
	bool	isSameMinute( const Date& date ) const	{ return mYear+mMonth+mDay+mHour+mMinute == date.mYear + date.mMonth + date.mDay + date.mHour + date.mMinute; }

	// -----------------------------------------------------------------
	//		変数		
	// -----------------------------------------------------------------
	u16		mYear;		//!< 年
	u16		mMonth;		//!< 月
	u16		mDayOfWeek;	//!< 曜日（SYSTEMTIME互換用）
	u16		mDay;		//!< 日
	u16		mHour;		//!< 時
	u16		mMinute;	//!< 分
	u16		mSecond;	//!< 秒
	u16		mMillisec;	//!< ミリ秒
};
}	// TGL

/* ========================================================================= */
/*		EOF																	 */
/* ========================================================================= */
