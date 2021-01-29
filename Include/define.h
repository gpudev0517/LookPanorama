#ifndef DEFINE_H
#define DEFINE_H
#include <QDebug>
#include <QTime>
#include <qfile.h>
#include "PanoLog.h"

#define THREAD_TYPE_ID				1000
#define THREAD_TYPE_MAIN			THREAD_TYPE_ID + 0
#define THREAD_TYPE_STITCHER		THREAD_TYPE_ID + 1
#define THREAD_TYPE_CAPTURE			THREAD_TYPE_ID + 2
#define THREAD_TYPE_STREAM			THREAD_TYPE_ID + 3
#define THREAD_TYPE_AUDIO			THREAD_TYPE_ID + 5
#define THREAD_TYPE_OCULUS			THREAD_TYPE_ID + 6
#define THREAD_TYPE_BANNER			THREAD_TYPE_ID + 7
#define THREAD_TYPE_PLAYBACK		THREAD_TYPE_ID + 8
#define THREAD_TYPE_LIDAR			THREAD_TYPE_ID + 9

#define MSG_OCULUS_SWITCHOFF		"Switch off Oculus Rift"

#define PLAYBACK_CAMERA_INDEX				-1
#define NODAL_CAMERA_INDEX			-20

// range of nodal deviceNumber is [NODAL_CAMERA_INDEX, NODAL_CAMERA_INDEX + 10]
#define IS_NODAL_CAMERA_INDEX(id) (id >= NODAL_CAMERA_INDEX && id < NODAL_CAMERA_INDEX + 10)

#define CUR_TIME							QTime::currentTime().toString("mm:ss.zzz")
#define CUR_TIME_H							QTime::currentTime().toString("hh.mm.ss")
#define ARGN(num)							arg(QString::number(num))
#define LUT_COUNT 11

#define PANO_CONN(sender, signal, receiver, slot)	connect(sender, SIGNAL(signal), receiver, SLOT(slot))

extern PanoLog* g_logger;

//#define PANO_LOG(msg)			qDebug() << QString("[%1] %2").arg(this->m_Name).arg(msg)
#define PANO_LOG(msg)			if(g_logger) g_logger->info(QString("[%1] %2").arg(this->m_Name).arg(msg))
#define PANO_LOG_ARG(msg, arg1)	if(g_logger) g_logger->info(QString(QString("[%1] %2").arg(this->m_Name).arg(msg)).arg(arg1))
#define PANO_N_LOG(msg)			if(g_logger) g_logger->info(msg, true)
#define PANO_DEVICE_LOG(msg)	if(g_logger) g_logger->info(QString("[%1] [%2] %3").arg(this->m_Name).arg(QString::number(m_deviceNumber)).arg(msg))
#define PANO_DEVICE_N_LOG(msg)	if(g_logger) g_logger->info(msg, true)
#define PANO_WARN(msg)			if(g_logger) g_logger->warning(QString("[%1] %2").arg(this->m_Name).arg(msg))
#define PANO_WARN_ARG(msg, arg1) if(g_logger) g_logger->warning(QString(QString("[%1] %2").arg(this->m_Name).arg(msg)).arg(arg1))
#define PANO_N_WARN(msg)		if(g_logger) g_logger->warning(msg, true)
#define PANO_DEVICE_WARN(msg)	if(g_logger) g_logger->warning(QString("[%1] [%2] %3").arg(this->m_Name).arg(QString::number(m_deviceNumber)).arg(msg))
#define PANO_DEVICE_N_WARN(msg)	if(g_logger) g_logger->warning(msg, true)
#define PANO_ERROR(msg)			if(g_logger) g_logger->critical(QString("[%1] %2").arg(this->m_Name).arg(msg))
#define PANO_N_ERROR(msg)		if(g_logger) g_logger->critical(msg, true)
#define PANO_DEVICE_ERROR(msg)	if(g_logger) g_logger->critical(QString("[%1] [%2] %3").arg(this->m_Name).arg(QString::number(m_deviceNumber)).arg(msg))
#define PANO_DEVICE_N_ERROR(msg)	if(g_logger) g_logger->critical(msg, true)
#define PANO_DLOG(msg)			if(g_logger) g_logger->debug(QString("[%1] %2").arg(this->m_Name).arg(msg))
#define PANO_N_DLOG(msg)		if(g_logger) g_logger->debug(msg, true)
#define PANO_DEVICE_DLOG(msg)	if(g_logger) g_logger->debug(QString("[%1] [%2] %3").arg(this->m_Name).arg(QString::number(m_deviceNumber)).arg(msg))
#define PANO_DEVICE_N_DLOG(msg)	if(g_logger) g_logger->debug(msg, true)

#define PANO_TIME_LOG(msg)			if(g_logger) g_logger->info( QString("[%1] %2 (%3)").arg(this->m_Name).arg(msg).arg(CUR_TIME))
#define PANO_DEVICE_TIME_LOG(msg)	if(g_logger) g_logger->info(QString("[%1] [%2] %3 (%4)").arg(this->m_Name).arg(QString::number(m_deviceNumber)).arg(msg).arg(CUR_TIME))

#if 0
//#define PANO_DEBUG
#ifdef _DEBUG
#define PANO_DLOG(msg)			g_logger->debug(QString("[%1] %2").arg(this->m_Name).arg(msg))
#define PANO_DEVICE_DLOG(msg)	g_logger->debug(QString("[%1] [%2] %3").arg(this->m_Name).arg(QString::number(m_deviceNumber)).arg(msg))
#else
#define PANO_DLOG(msg)			{}
#define PANO_DEVICE_DLOG(msg)	{}
#define PANO_TIME_DLOG(msg)		{}
#define PANO_DEVICE_TIME_DLOG(msg)		{}
#endif
#endif

#define ERROR_MSG(msg)						QMessageBox::critical(this, tr("Error"), msg)
#define WARNING_MSG(msg)					QMessageBox::warning(this, tr("Warning"), msg)
//#define LOG(msg)							qDebug() << msg//.toStdString().c_str())
#define Q_LOG(msg)							T_LOG(msg.toStdString().c_str())
#define FREE_PTR(ptr)						{ if (ptr) { delete ptr; ptr = NULL; } }
#define FREE_MEM(ptr)						{ if (ptr) { free(ptr); ptr = NULL; } }
#define FREE_AV_MEM(ptr)						{ if (*ptr) { av_frame_free(ptr); *ptr = NULL; } }

typedef unsigned char	byte;

#define LIDAR_HEIGHT 64
#define LIDAR_WIDTH 1024
#define LIDAR_STREAM_HEIGHT 2048
#define LIDAR_STREAM_WIDTH 1024
#define LIDER_BAND_HEIGHT 80  //64 + 16
#define LIDAR_STREAM_MAX_VALUE 200
#define LIDAR_RANGE_RESOLUTION 1.2  //cm
#define LIDAR_BAND_COUNT 25   // LIDAR_STREAM_HEIGHT : LIDER_BAND_HEIGHT

#endif // DEFINE_H