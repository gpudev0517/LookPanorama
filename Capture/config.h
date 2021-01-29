#ifndef SCY_WebRTC_WebRTCStreamer_Config_H
#define SCY_WebRTC_WebRTCStreamer_Config_H


#include "scy/base.h"

#define SERVER_HOST "localhost"
//#define SERVER_HOST "10.70.3.52"
//#define SERVER_HOST "10.70.3.60"
//#define SERVER_HOST "richard_song-pc"
#define USE_SSL 0 // 1
#if USE_SSL
#define SERVER_PORT 443
#else
#define SERVER_PORT 4500
#endif


#endif
