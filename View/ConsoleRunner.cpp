#include "ConsoleRunner.h"

extern CoreEngine* g_Engine;

ConsoleRunner::ConsoleRunner()
{
}

ConsoleRunner::~ConsoleRunner()
{
}

bool ConsoleRunner::initPaths(QString configpath, QString broadcasturl)
{

	m_iniPath = configpath;
	m_streamPath = broadcasturl;
	return true;
}

void ConsoleRunner::openIniPath( QString iniPath )
{
	g_Engine->openIniPath( iniPath );
}

void ConsoleRunner::openProject( QString iniPath )
{
 	openIniPath( iniPath );
 	g_Engine->openProject();
}

void ConsoleRunner::startStream( QString streamPath, int width, int height, int streamMode )
{
	g_Engine->setPlayMode( PlayMode::PAUSE_MODE );
	g_Engine->startStreaming( streamPath, width, height, streamMode );
}
void ConsoleRunner::run()
{
	connect( g_Engine, SIGNAL( started( bool ) ), this, SLOT( startStream( bool ) ) );
	openProject( m_iniPath );
}
void ConsoleRunner::startStream(bool bFinish)
{
	if(bFinish)
		startStream( m_streamPath, 1280, 720, StreamingMode::RTMP );
}