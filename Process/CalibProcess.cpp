#include "CalibProcess.h"
#include "define.h"

#include <QFile>
#include <QDir>
#include <QThread>

#include "SharedImageBuffer.h"
#include "QmlMainWindow.h"

static QString curBlock = ROOT_BLOCK;
static int curImageIndex = 0, ctIndex = 0, optIndex = 0;

extern QmlMainWindow* g_mainWindow;

CalibProcess::CalibProcess(QString command, QObject * parent)
    : QObject(parent)
    , m_Name("CalibProcess")
    , m_Output("")
	, m_isError(false)
	, m_isFinished(false)
	, m_isFinal(false)
{
	if (!command.isNull() && !command.isEmpty())
		setCommand(command);

    PANO_CONN(&m_Process, readyReadStandardOutput(), this, readOutput());
    PANO_CONN(&m_Process, finished(int), this, finishedCalib());
	PANO_CONN(&m_Process, error(QProcess::ProcessError), this, errorProcess(QProcess::ProcessError));

	if (parent)
		PANO_CONN(this, finishedSignal(int), parent, calibrate(int));
}

void CalibProcess::initialize()
{
	QDir curPath(".");
	QStringList fileList = QDir(".").entryList(QDir::Files);
	foreach(const QString file, fileList) {
		if (!(file.endsWith(".pto") || file.endsWith(".jpg") || file.endsWith(".png") || file.endsWith(".bmp")))
			continue;
		QFile::remove(file);
	}
}

void CalibProcess::setCommand(QString command)
{
	if (command.isNull() || command.isEmpty()) {
		PANO_LOG("Command is invalid!");
		return;
	}

	QStringList args = command.split(" ");
	m_Process.setProgram(CALIB_WORK_PATH + args[0]);

	if (args.length() == 0)		return;

	m_Process.setArguments(QStringList());

	for (int i = 1; i < args.length(); i++) {
		if (args.at(i).isNull() || args.at(i).isEmpty())
			continue;

		QStringList orgArgs = m_Process.arguments();
		orgArgs.append(args.at(i));
		m_Process.setArguments(orgArgs);
	}
}

QString CalibProcess::getCommand()
{
	QString command = m_Process.program();
	QStringList args = m_Process.arguments();
	if (args.length() == 0)
		return command;

	foreach(const QString arg, args) {
		command += " " + arg;
	}

	return command;
}

void CalibProcess::addArg(QString arg)
{
    if (arg.isNull() || arg.isEmpty())  return;

    QStringList args = m_Process.arguments();
    args.append(arg);
    m_Process.setArguments(args);
}

bool CalibProcess::calibrate(QString command, bool isFinal)
{
    if (command != "") {
        setCommand(command);
    } else if (m_Process.program() == "") {
        PANO_LOG("Command empty!");
        return false;
    }

	m_isFinal = isFinal;

    PANO_LOG("Calibrating...");
	PANO_LOG(getCommand());

    m_Output.clear();

    m_Process.start();
    if (!m_Process.waitForStarted()) {
        PANO_LOG(QString("Command can not be start! (%1)").arg(m_Process.program()));
        return false;
    }

	QThread::msleep(100);		// This code must be need. If commented, following error opened.

    return true;
}

void CalibProcess::errorProcess(QProcess::ProcessError error)
{
	PANO_LOG(QString("Calibrating failed! (errorCode:%1)").arg(error));
	m_isError = true;
}

void CalibProcess::readOutput()
{
    //PANO_DLOG(QString("Reading standard output string... (%1 bytes)").ARGN(m_Process.bytesAvailable()));

	int availiable = m_Process.bytesAvailable();
	if (availiable > 0) {
		QString curRead(m_Process.readAll());
		if (m_Output.isNull() || m_Output.isEmpty())
			m_Output = curRead;
		else
			m_Output += curRead;
	}
}

void CalibProcess::finishedCalib()
{
    PANO_LOG("Finished calibrating.");
#if 0
	if (!m_Output.isNull() && !m_Output.isEmpty()) {
		QStringList resultList = m_Output.split("\r\n");

		foreach(const QString line, resultList) {
			if (line == "") continue;
			qDebug() << line;
		}
	}
#endif
    
	if (!m_isError && m_isFinal)	{
		//parsePTO(m_Process.workingDirectory() + "/" + CALIB_RESULT_FILE);
		curBlock = ROOT_BLOCK;
		curImageIndex = 0;
		ctIndex = 0;
		optIndex = 0;
		QMap<QString, PTOVAL> paramMap;
		if (!parsePTO(CALIB_RESULT_FILE, true, paramMap))
		{
			parsePTO(CALIB_RESULT_FILE_2, false, paramMap);
		}
		m_isFinished = true;
		emit finishedSignal(99);
	} else
		emit finishedSignal(-1);
}

bool CalibProcess::parsePTO(QString filename, bool isPrimary, QMap<QString, PTOVAL>& paramMap)
{
    QString ptoFileName = "";

    if (filename.isEmpty() || filename.trimmed() == "") {
        ptoFileName = m_Process.workingDirectory() + "/" + CALIB_RESULT_FILE;
    } else
        ptoFileName = filename;

    QFile ptoFile(ptoFileName);

    if (!ptoFile.exists()) {
		if (isPrimary)
		{
			qDebug() << ptoFileName << " is not exist! Maybe due to color adjustment failed.";
		}
		else
		{
			qDebug() << ptoFileName << " is not exist! Camera calibration failed.";
		}
		paramMap = m_ptoConfig;
		return false;
    }

    if (ptoFile.size() < 1024) {
        qDebug() << "File name is invalid! (Size is small)";
		paramMap = m_ptoConfig;
		return false;
    }

    if (!ptoFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "\"" << ptoFileName << "\" is not text file!";
    }

    m_ptoConfig.clear();

    while (!ptoFile.atEnd()) {
        QString line = ptoFile.readLine();
        processLine(line);
    }
/*
    int imgIndex = 0;
    while (m_ptoConfig.contains(QString("%1%2").arg(IMAGE_KEY).arg(QString::number(imgIndex)))) {
        PTOVAL entry = m_ptoConfig[QString("%1%2").arg(IMAGE_KEY).arg(QString::number(imgIndex))];
        //qDebug() << endl << entry.value;
        QMap<QString, float> params = entry.camParams;
        QMap<QString, float> shareParams = m_ptoConfig[SHARE_KEY].camParams;
        QMapIterator<QString, QString> i(m_pacPTS);
        while (i.hasNext()) {
            i.next();
            QString key = i.value();
            float value = 0;
            if (shareParams.contains(key)) {
                value = shareParams[key];
            }
            if ((shareParams.contains(key) && value == 0) && params.contains(key)) {
                value = params[key];
            }
            if (key == EXPOSURE_VAL && m_ptoConfig.contains(EXPOSURE_KEY + QString::number(imgIndex))) {
                QString expOffsetStr = m_ptoConfig[EXPOSURE_KEY + QString::number(imgIndex)].value.at(0);
                value = expOffsetStr.toFloat();
            }
            //qDebug() << QString("%1 (%2) = %3").arg(i.key()).arg(key).arg(value);
        }
        imgIndex++;
    }*/
	paramMap = m_ptoConfig;
    return true;
}

int CalibProcess::processLine(QString line)
{
    QString firstCh = line.left(2);

    if (firstCh == COMMENT_EXP) {    // Start Block
        QString block = line.mid(2, line.length()-3);
        qDebug() << "Start Block: " << block;
        curBlock = block;
        return 0;
    }

    if (line.startsWith(VALUE_EXP)) {                             // Value
        QString valStr = line.mid(1, line.length() - 2);
        QStringList tokens = valStr.split(" ");
        QString valName = tokens.first();

        TYPEOFVAL valType = NONE_VAL;

        tokens.removeAt(0);
        if (tokens.length() == 0)   valType = BOOL_VAL;
        else if (tokens.length() == 1) {
            QString value = tokens.at(0);
            bool flag1 = false, flag2 = false;
            value.toFloat(&flag1);
            value.toInt(&flag2);
            if (flag1 || flag2)
                valType = NUMBER_VAL;
            else
                valType = STRING_VAL;
        } else {
            valType = STRUCTURE_VAL;
        }

        QString key = curBlock == CP_VAL ? ROOT_BLOCK : curBlock + "->" + valName;
        PTOVAL entry = {valType, tokens, tokens.length()};

        QString log = QString("key=%1, type=%2, value=%3").arg(key).arg(QString::number(valType)).arg(tokens.join(" "));
        //qDebug() << log;
        m_ptoConfig[key] = entry;
    }

    QString key;

    // ex: i w1920 h1080 f3 v239.783274799023 Ra0 Rb0 Rc0 Rd0 Re0 Eev0 Er1 Eb1 r0 p0 y0 TrX0 TrY0 TrZ0 Tpy0 Tpp0 j0 a-0.0289964935407136 b0.0695966256514294 c-0.0429056763580283 d-29.9061717662073 e-86.0841666213265 g0 t0 Va1 Vb0 Vc0 Vd0 Vx0 Vy0  Vm5 n"office-00.jpg"
    if (firstCh == CAMERA_EXP) {
        QString valStr = line.mid(2, line.length() - 3);
        QStringList tokens = valStr.split(" ");
        QMap<QString, float> valEnties;
        for (int i = 0; i < tokens.length(); i++) {
            QString exp = tokens.at(i);
            if (exp.isEmpty())  continue;
            QString name = exp.left(1).toLower();
            QString val = "";
            if (exp.length() > 2 && exp.mid(1,1) == "=") {      // x=y
                val = exp.right(exp.length() - 2);
			} else if (exp.length() > 3 && exp.mid(0, 1) >= "A" && exp.mid(0, 1) <= "Z" && exp.mid(2, 1) >= "a" && exp.mid(2, 1) <= "z") {    // Eev-0.421924633629934
				name = exp.left(3);
				val = exp.right(exp.length() - 3);
			} else if (exp.length() > 2 && exp.mid(0, 1) >= "A" && exp.mid(0, 1) <= "Z") {    // Ra0
				name = exp.left(2);
				val = exp.right(exp.length() - 2);
            } else if (exp.length() > 3 && exp.mid(1,1) == "\"" && exp.right(1) == "\"") {    // n"office-01.jpg"
                val = exp.mid(2, exp.length() - 3);
            }
            else                                              // y86.61479670316621
                val = exp.right(exp.length() - 1);

            bool fflag = false;
            float fVal = val.toFloat(&fflag);
            if (fflag == true)
                valEnties[name] = fVal;
            //qDebug() << QString("%1=%2").arg(name).arg(name=="n" ? val : QString::number(fVal));
        }
        key = IMAGE_VAL + QString::number(curImageIndex++);
        PTOVAL entry = m_ptoConfig[key];
        if (valEnties.size() > 0)
            entry.camParams = valEnties;
        m_ptoConfig[key] = entry;
        //qDebug() << QString("key=%1, camParams=%2").arg(key).arg(valStr);
    }
	// c n0 N6 x361.429992867723 y417.454743642619 X665.802607065699 Y604.530152387064 t0
    else if (firstCh == CP_EXP) {
        QString valStr = line.mid(2, line.length() - 3);
        QStringList tokens = valStr.split(" ");
        QMap<QString, float> valEnties;
        for (int i = 0; i < tokens.length(); i++) {
            QString exp = tokens.at(i);
            if (exp.isEmpty())  continue;
            QString name = exp.left(1);
            // y86.61479670316621
            QString val = exp.right(exp.length() - 1);

            bool fflag = false;
            float fVal = val.toFloat(&fflag);
            if (fflag == true)
                valEnties[name] = fVal;
            //qDebug() << QString("%1=%2").arg(name).arg(name=="n" ? val : QString::number(fVal));
        }
        key = CP_VAL + QString::number(ctIndex++);
        PTOVAL entry = m_ptoConfig[key];
        if (valEnties.size() > 0)
            entry.camParams = valEnties;
        m_ptoConfig[key] = entry;
        //qDebug() << QString("key=%1, controlPoint=%2").arg(key).arg(valStr);
    }
    else if (firstCh == OPT_EXP) {
        QString valStr = line.mid(2, line.length() - 3);
        QStringList tokens = valStr.split(" ");
        QMap<QString, float> valEnties;
        for (int i = 0; i < tokens.length(); i++) {
            QString exp = tokens.at(i);
            if (exp.isEmpty())  continue;
            QString name = exp.left(1);
            // y86.61479670316621
            QString val = exp.right(exp.length() - 1);

            bool fflag = false;
            float fVal = val.toFloat(&fflag);
            if (fflag == true)
                valEnties[name] = fVal;
            //qDebug() << QString("%1=%2").arg(name).arg(name=="n" ? val : QString::number(fVal));
        }
        key = OPT_VAL + QString::number(optIndex++);
        PTOVAL entry = m_ptoConfig[key];
        if (valEnties.size() > 0)
            entry.camParams = valEnties;
        m_ptoConfig[key] = entry;
        //qDebug() << QString("key=%1, optimized=%2").arg(key).arg(valStr);
    }

    return 0;
}

// SingleCameraCalibProcess

SingleCameraCalibProcess::SingleCameraCalibProcess()
: previewBuffer(0)
, m_threadInstance(0)
{
	detector.setVerifier(&solver);
	connect(&detector, SIGNAL(fireSnapshotChanged(int)), this, SLOT(onSnapshot(int)));
	connect(&detector, SIGNAL(strengthRatioChanged(float)), this, SLOT(onStrengthRatio(float)));
	connect(&detector, SIGNAL(strengthDrawColorChanged(QString)), this, SLOT(onStrengthDrawColor(QString)));

	nSnapshots = 0;

	doStop = false;
	m_finished = true;

	isSrcReady = false;
	isOnCalibration = false;
	startThread();
}
SingleCameraCalibProcess::~SingleCameraCalibProcess()
{
	qquit();
	disconnect(&detector, SIGNAL(fireSnapshotChanged(int)), this, SLOT(onSnapshot(int)));
	disconnect(&detector, SIGNAL(strengthRatioChanged(float)), this, SLOT(onStrengthRatio(float)));
	disconnect(&detector, SIGNAL(strengthDrawColorChanged(QString)), this, SLOT(onStrengthDrawColor(QString)));

	if (previewBuffer)
	{
		delete[] previewBuffer;
		previewBuffer = NULL;
	}
}

void SingleCameraCalibProcess::setSharedImageBuffer(SharedImageBuffer *sharedImageBuffer)
{
	this->sharedImageBuffer = sharedImageBuffer;
}

void SingleCameraCalibProcess::initialize()
{
	detector.clear();
	nSnapshots = 0;
	calibStatus = None;

	renderStatus();

	int imgWidth = sharedImageBuffer->getGlobalAnimSettings()->cameraSettingsList()[0].xres;
	int imgHeight = sharedImageBuffer->getGlobalAnimSettings()->cameraSettingsList()[0].yres;

	if (previewBuffer)
	{
		delete[] previewBuffer;
		previewBuffer = NULL;
	}
	previewBuffer = new unsigned char[imgWidth * imgHeight * 3];
}

void SingleCameraCalibProcess::setParameter(int camIndex, CameraParameters::LensType lensType,
	int boardSizeW, int boardSizeH, int snapshotNumber)
{
	this->camIndex = camIndex;
	this->maxSnapshotCount = snapshotNumber;
	solver.setBoardSize(boardSizeW, boardSizeH);
	solver.setLensType(lensType == CameraParameters::LensType_opencvLens_Fisheye);
}

void SingleCameraCalibProcess::startCapture()
{
	nSnapshots = 0;
	detector.clear();

	calibStatus = Capturing;
}

void SingleCameraCalibProcess::takeSnapshot() {
	//isTakeSnapshoting = true;
	//calibStatus = TakeSnapshoting;
}
void SingleCameraCalibProcess::stopCapture()
{
	calibStatus = None;
	emit fireSnapshotFinished();
	if (isReadyToCalibrate())
	{
		calibStatus = ReadyToCalibrate;
	}
}

bool SingleCameraCalibProcess::isReadyToCalibrate()
{
	if (nSnapshots >= 4)
		return true;
	return false;
}

bool SingleCameraCalibProcess::calibrate()
{
	if (isReadyToCalibrate())
	{
		threadMutex.lock();
		isOnCalibration = true;
		threadMutex.unlock();

		return true;
	}
	else
	{
		return false;
	}
}

void SingleCameraCalibProcess::apply()
{
	sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(camIndex).m_cameraParams = solver.getCameraParams();
	sharedImageBuffer->getGlobalAnimSettings()->m_lensType = sharedImageBuffer->getGlobalAnimSettings()->getCameraInput(camIndex).m_cameraParams.m_lensType;
	sharedImageBuffer->getStitcher().get()->restitch(true);
}

void SingleCameraCalibProcess::onFrame(unsigned char* buffer, int width, int height)
{
	if (calibStatus != Capturing) return;

	Mat img(height, width, CV_8UC3, buffer, Mat::AUTO_STEP);
	
	threadMutex.lock();
	cv::flip(img, srcImg, 0);
	isSrcReady = true;
	threadMutex.unlock();
}

void SingleCameraCalibProcess::onSnapshot(int snapshotCount)
{
	nSnapshots = snapshotCount;
	if (snapshotCount >= maxSnapshotCount)
	{
		stopCapture();
		bool isStopedCapture = false;
		emit g_mainWindow->stopSingleCaptureChanged(isStopedCapture);
	}
}

void SingleCameraCalibProcess::onStrengthRatio(float strengthRatio) {

	g_mainWindow->sendStrengthRatioChanged(strengthRatio);
}

void SingleCameraCalibProcess::onStrengthDrawColor(QString color) {
	g_mainWindow->sendStrengthDrawColorChanged(color);
}

void SingleCameraCalibProcess::renderStatus()
{
	std::string msg;
	switch (calibStatus)
	{
	case None:
		msg = format("Press Start Capture to start");
		break;
	case Capturing:
		msg = format("Capturing: %d/%d", detector.getSnapshots().size(), maxSnapshotCount);
		break;
	case ReadyToCalibrate:
		msg = format("Press Calibrate.");
		break;
	case FindCorners:
		msg = format("Finding corners.");
		break;
	case Calibrating:
		msg = format("Calibrating...");
		break;
	case Finished:
		msg = "Calibration sucess. avg reprojection error = " + std::to_string(m_avgErr);
		break;
	case Failed:
		msg = format("Calibration failed.");
		break;
	}
	

	g_mainWindow->sendSingleCalibMessageChanged(QString::fromStdString(msg));
	g_mainWindow->sendSingleCalibStatusChanged(calibStatus);

}

void SingleCameraCalibProcess::process()
{
	m_finished = false;
	run();
	finishWC.wakeAll();
}

void SingleCameraCalibProcess::qquit()
{
	if (m_threadInstance && m_threadInstance->isRunning())
		stopThread();
	releaseThread();
}

void SingleCameraCalibProcess::setupThread()
{
	releaseThread();

	m_threadInstance = new QThread;
	this->moveToThread(m_threadInstance);
	connect(m_threadInstance, SIGNAL(started()), this, SLOT(process()));
}

void SingleCameraCalibProcess::startThread()
{
	setupThread();
	m_threadInstance->start();
}

void SingleCameraCalibProcess::stopThread()
{
	QMutexLocker locker(&doStopMutex);
	doStop = true;
	//sharedImageBuffer->wakeStitcher();
}

void SingleCameraCalibProcess::waitForFinish()
{
	finishMutex.lock();
	finishWC.wait(&finishMutex, 100);
	finishMutex.unlock();
}

void SingleCameraCalibProcess::setFinished()
{
	m_finished = true;
}

bool SingleCameraCalibProcess::isFinished()
{
	return m_finished;
}

void SingleCameraCalibProcess::releaseThread()
{
	if (m_threadInstance)
	{
		m_threadInstance->quit();
		m_threadInstance->wait();
		delete m_threadInstance;
		m_threadInstance = NULL;
	}
}

void SingleCameraCalibProcess::run()
{
	while (1)
	{
		if (QThread::currentThread()->isInterruptionRequested())
		{
			doStop = true;
		}

		//
		// Stop thread if doStop = TRUE 
		//
		doStopMutex.lock();
		if (doStop)
		{
			doStop = false;
			doStopMutex.unlock();
			break;
		}
		doStopMutex.unlock();

		// detector
		bool isDetector = false;
		int width, height;
		threadMutex.lock();
		if (isSrcReady)
		{
			if (calibStatus == Capturing)
			{
				detector.detect(srcImg);
			}

			isSrcReady = false;
			isDetector = true;
		}
		threadMutex.unlock();

		if (isDetector)
		{
			renderStatus();
		}

		// calibration
		bool isToCalibrate = false;
		threadMutex.lock();
		if (isOnCalibration)
		{
			isToCalibrate = true;
			isOnCalibration = false;
		}
		threadMutex.unlock();

		if (isToCalibrate)
		{
			std::vector<Mat> snapshots = detector.getSnapshots();
			calibStatus = Calibrating;

			renderStatus();
			double avgErr = 0;
			if (solver.calibrate(snapshots, previewBuffer, avgErr))
			{
				m_avgErr = avgErr;
				apply();
				calibStatus = Finished;
			}
			else
			{
				calibStatus = Failed;
			}

			renderStatus();			
			//ImageBufferData frame = getSelfCalibFrame();
			//D360Stitcher* stitcher = sharedImageBuffer->getStitcher().get();
			//stitcher->updateCalibPreviewImage(frame);
		}
	}
}
