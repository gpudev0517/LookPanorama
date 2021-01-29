#ifndef	_TGLSYSTEMTYPES_H_
#define	_TGLSYSTEMTYPES_H_
/* ========================================================================= */
/*		定義																 */
/* ========================================================================= */
//!	SYSINFO_T.uFlag
#define	SYSINFOF_FIRSTBOOT	(1<< 0)	//!< 初回起動
#define	SYSINFOF_CREATEINI	(1<< 1)	//!< 設定ファイル生成
#define	SYSINFOF_FULLSCREEN	(1<< 2)	//!< フルスクリーンモード
#define	SYSINFOF_VSYNC		(1<< 3)	//!< VSYNC有効
#define	SYSINFOF_PADPOVMAIN	(1<< 4)	//!< パッドのPOVを主軸と同期させる
#define	SYSINFOF_PADVIB		(1<< 5)	//!< パッド振動
#define	SYSINFOF_AEROOFF	(1<< 6)	//!< WindowsAeroを無効にする
#define	SYSINFOF_SFX		(1<< 7)	//!< SFX有効
#define	SYSINFOF_ANAGLYPH	(1<< 8)	//!< 立体視 
#define	SYSINFOF_NOSURROUND	(1<< 9)	//!< サラウンド無効
#define	SYSINFOF_LFE		(1<<10)	//!< サウンドLFE有効
#define	SYSINFOF_BGEXEC		(1<<11)	//!< バックグラウンド実行
#define	SYSINFOF_12			(1<<12)	//!< 
#define	SYSINFOF_13			(1<<13)	//!< 
#define	SYSINFOF_14			(1<<14)	//!< 
#define	SYSINFOF_15			(1<<15)	//!< 
#define	SYSINFOF_16			(1<<16)	//!< 
#define	SYSINFOF_17			(1<<17)	//!< 
#define	SYSINFOF_18			(1<<18)	//!< 
#define	SYSINFOF_19			(1<<19)	//!< 
#define	SYSINFOF_20			(1<<20)	//!< 
#define	SYSINFOF_21			(1<<21)	//!< 
#define	SYSINFOF_22			(1<<22)	//!< 
#define	SYSINFOF_23			(1<<23)	//!< 
#define	SYSINFOF_24			(1<<24)	//!< 
#define	SYSINFOF_25			(1<<25)	//!< 
#define	SYSINFOF_26			(1<<26)	//!< 
#define	SYSINFOF_27			(1<<27)	//!< 
#define	SYSINFOF_28			(1<<28)	//!< 
#define	SYSINFOF_29			(1<<29)	//!< 
#define	SYSINFOF_30			(1<<30)	//!< 
#define	SYSINFOF_31			(1<<31)	//!< 

//!	OS
#define	SYSINFOOS_UNKNOWN		(0)		//!< 不明
#define	SYSINFOOS_WINDOWS		(1)		//!< Windowsシリーズの何か
#define	SYSINFOOS_WINDOWS_95	(2)		//!< 95
#define	SYSINFOOS_WINDOWS_98	(3)		//!< 98
#define	SYSINFOOS_WINDOWS_ME	(4)		//!< Me
#define	SYSINFOOS_WINDOWS_9X	(5)		//!< 9x
#define	SYSINFOOS_WINDOWS_3_X	(6)		//!< 3.x
#define	SYSINFOOS_WINDOWS_NT4	(7)		//!< NT 4.0
#define	SYSINFOOS_WINDOWS_2000	(8)		//!< 2000
#define	SYSINFOOS_WINDOWS_XP	(9)		//!< XP
#define	SYSINFOOS_WINDOWS_S2003	(10)	//!< Server2003
#define	SYSINFOOS_WINDOWS_VISTA	(11)	//!< Vista
#define	SYSINFOOS_WINDOWS_7		(12)	//!< 7
#define	SYSINFOOS_WINDOWS_8		(13)	//!< 8
#define	SYSINFOOS_MAX			(14)	//!< 

//!	解像度
#define	SYSINFORESO_320_240		(0)	//!< 4:3		
#define	SYSINFORESO_640_480 	(1)
#define	SYSINFORESO_800_600 	(2)
#define	SYSINFORESO_1024_768	(3)
#define	SYSINFORESO_1280_960	(4)
#define	SYSINFORESO_1600_1200	(5)
#define	SYSINFORESO_640_360		(6)	//!< 16:9		
#define	SYSINFORESO_848_480		(7)
#define	SYSINFORESO_1280_720	(8)
#define	SYSINFORESO_1600_900	(9)
#define	SYSINFORESO_1920_1080	(10)
#define	SYSINFORESO_2560_1440	(11)
#define	SYSINFORESO_3840_2160	(12) //!< 4K
#define	SYSINFORESO_7680_4320	(13) //!< 8K
#define	SYSINFORESO_MAX			(14)
//! 16:9開始
#define	SYSINFORESO_16_9_START	SYSINFORESO_640_360
//!	過大サイズ
#define	SYSINFORESO_LARGE_START	SYSINFORESO_2560_1440

//!	アスペクト比
#define	SYSINFOASPECT_NONE	(0)	//!< 無効
#define	SYSINFOASPECT_4_3	(1)	//!< 4:3
#define	SYSINFOASPECT_5_4	(2)	//!< 5:4
#define	SYSINFOASPECT_15_9	(3)	//!< 15:9
#define	SYSINFOASPECT_16_9	(4)	//!< 16:9
#define	SYSINFOASPECT_16_10	(5)	//!< 16:10
#define	SYSINFOASPECT_MAX	(6)	//!< 

//!	MSAA
#define	SYSINFOMSAA_NONE	(0)	//!< 無効
#define	SYSINFOMSAA_X2		(1)	//!< x2
#define	SYSINFOMSAA_X4		(2)	//!< x4
#define	SYSINFOMSAA_X6		(3)	//!< x6
#define	SYSINFOMSAA_X8		(4)	//!< x8
#define	SYSINFOMSAA_X16		(5)	//!< x16
#define	SYSINFOMSAA_MAX		(6)	//!< 

//!	アンチエイリアス
#define	SYSINFOAA_NONE			(0)	//!< 無効
#define	SYSINFOAA_FXAA			(1)	//!< FXAA
#define	SYSINFOAA_SMAA_LOW		(2)	//!< SMAA
#define	SYSINFOAA_SMAA_MID		(3)	//!< SMAA
#define	SYSINFOAA_SMAA_HIGH		(4)	//!< SMAA
#define	SYSINFOAA_SMAA_ULTRA	(5)	//!< SMAA
#define	SYSINFOAA_MAX			(6)

//!	テクスチャフィルタ
#define SYSINFOTEXF_POINT		(0)	//!< 補間無し
#define SYSINFOTEXF_LINEAR		(1)	//!< 線形
#define SYSINFOTEXF_ANISOTROPIC	(2)	//!< 異方正
#define SYSINFOTEXF_MAX			(3)

//!	ポストフィルタ品質
#define SYSINFOPFLV_STD		(0)	//!< 標準
#define SYSINFOPFLV_LOW		(1)	//!< 低品質
#define SYSINFOPFLV_MAX		(2)

//!	グラフィックスレベル
#define SYSINFOGRPLV_STD	(0)	//!< 標準
#define SYSINFOGRPLV_LOW	(1)	//!< 低品質
#define SYSINFOGRPLV_MAX	(2)

//!	ポストフィルタ品質
#define SYSINFOGRPLV_STD	(0)	//!< 標準
#define SYSINFOGRPLV_LOW	(1)	//!< 低品質
#define SYSINFOGRPLV_MAX	(2)

//!	プリセット品質
#define SYSINFOPQ_LOW		(0)	//!< 低品質
#define SYSINFOPQ_LOWMED	(1)	//!< 低中品質
#define SYSINFOPQ_MED		(2)	//!< 標準
#define SYSINFOPQ_HIGH		(3)	//!< 高品質
#define SYSINFOPQ_MAX		(4)

//!	リージョン
#define	SYSINFOREGION_UNKNOWN	(0)	//!< 不明
#define	SYSINFOREGION_JP		(1)	//!< 日本
#define	SYSINFOREGION_USA		(2)	//!< 北米
#define	SYSINFOREGION_EURO		(3)	//!< 欧州
#define	SYSINFOREGION_ASIA		(4)	//!< アジア
#define	SYSINFOREGION_MAX		(5)

//!	言語
#define	SYSINFOLANG_JP		(0)	//!< 日本語
#define	SYSINFOLANG_EN		(1)	//!< 英語
#define	SYSINFOLANG_MAX		(2)	//!< 


//!	FPSモード
#define	SYSINFOFPS_NONE		(0)	//!< 無効
#define	SYSINFOFPS_10		(1)	//!< 10
#define	SYSINFOFPS_20		(2)	//!< 20
#define	SYSINFOFPS_30		(3)	//!< 30
#define	SYSINFOFPS_50		(4)	//!< 50
#define	SYSINFOFPS_60		(5)	//!< 60
#define	SYSINFOFPS_120		(6)	//!< 120
#define	SYSINFOFPS_MAX		(7)	//!< 

#endif	/*	_TGLSYSTEMTYPES_H_	*/
//=============================================================================
// eof
//=============================================================================
