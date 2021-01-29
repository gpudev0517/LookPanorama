#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QtQml>
#include "MCQmlCameraView.h"
#include "QmlInteractiveView.h"
#include "D360Parser.h"
#include "QmlMainWindow.h"
#include "QmlRecentDialog.h"
#include "TakeMgrTreeModel.h"
#include "QmlTakeManagement.h"
#include "QmlApplicationSetting.h"
#include <QFileSystemModel>
#include "Pixmap.h"
#include "ConsoleRunner.h"

#if 0
#ifdef _DEBUG
#include <vld.h> 
#endif
#endif

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

QThread* g_mainThread;
CoreEngine *g_Engine = NULL;
int main(int argc, char *argv[])
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	qputenv("QML_USE_GLYPHCACHE_WORKAROUND", QByteArray("1"));
	QCoreApplication::addLibraryPath("./");
	QCoreApplication::setAttribute(Qt::AA_ShareOpenGLContexts, true);

	QGuiApplication *app = new QGuiApplication(argc, argv);
	app->setApplicationName("Look3D");
	app->setApplicationVersion("1.0.0");

	QCommandLineParser parser;
	parser.addHelpOption();
	parser.addVersionOption();
	QCommandLineOption consoleOption("nogui", QGuiApplication::translate("main", "Server mode"));
	parser.addOption(consoleOption);
	QCommandLineOption configOption(QStringList() << "c" << "config", QGuiApplication::translate("main", "config full filename"), QGuiApplication::translate("main", "file"));
	parser.addOption(configOption);
	QCommandLineOption broadcastOption(QStringList() << "b" << "broadcast", QGuiApplication::translate("main", "broadcast url"), QGuiApplication::translate("main", "url"));
	parser.addOption(broadcastOption);

	parser.process(*app);

	g_mainThread = app->thread();

	bool isConsole = parser.isSet(consoleOption);

	int ret = -1;

	//g_Engine = new CoreEngine();

	if (isConsole)
	{
		ConsoleRunner* runner = new ConsoleRunner();
		runner->initPaths(parser.value(configOption), parser.value(broadcastOption));

		g_Engine = new CoreEngine();
		runner->run();

		try
		{
			ret = app->exec();
		}
		catch (...)
		{

		}
	}
	else
	{
		if( qgetenv( "QT_QUICK_CONTROLS_STYLE" ).isEmpty() )
		{
			qputenv("QT_QUICK_CONTROLS_STYLE", "Flat");
		}

		TemplateModes::init();

		QQmlApplicationEngine engine;
		qmlRegisterType<MCQmlCameraView>("MCQmlCameraView", 1, 0, "MCQmlCameraView");
		qmlRegisterType<QmlInteractiveView>("QmlInteractiveView", 1, 0, "QmlInteractiveView");
		qmlRegisterType<QmlMainWindow>("QmlMainWindow", 1, 0, "QmlMainWindow");
		qmlRegisterType<QmlRecentDialog>("QmlRecentDialog", 1, 0, "QmlRecentDialog");
		qmlRegisterType<QmlTakeManagement>("QmlTakeManagement", 1, 0, "QmlTakeManagement");
		qmlRegisterType<TemplateModes>("Look3DQmlModule", 1, 0, "TemplateModes");
		qmlRegisterType<QmlApplicationSetting>("QmlApplicationSetting", 1, 0, "QmlApplicationSetting");
		qmlRegisterType<Pixmap>("Pixmap", 1, 0, "Pixmap");
		engine.addImageProvider("pixmap", new PixmapProvider);

		QStringList headers;
		headers << ("Session and Take");
		g_takeMgrModel = new TakeMgrTreeModel(headers, "");
		engine.rootContext()->setContextProperty("takeMgrModel", g_takeMgrModel);
		engine.load(QUrl("qrc:/main.qml"));

		try
		{
			ret = app->exec();
		}
		catch (...)
		{

		}
		delete g_takeMgrModel;
	}

	return ret;
}