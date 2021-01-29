#include "ptsparser.h"
#include "pts.h"

#include <QString>
#include <QFile>
#include <QDebug>

static QString curBlock = ROOT_BLOCK;
static int curImageIndex = 0;
static bool sharedValues = false;

PTSParser::PTSParser()
{
    // Initialize PAC and PTS map
    m_pacPTS["lesType"]     = "f";
    m_pacPTS["yaw"]         = "y";
    m_pacPTS["pitch"]       = "p";
    m_pacPTS["roll"]        = "r";
    m_pacPTS["fov"]         = "v";
    m_pacPTS["k1"]          = "a";
    m_pacPTS["k2"]          = "b";
    m_pacPTS["k3"]          = "c";
    m_pacPTS["offset_x"]    = "d";
    m_pacPTS["offset_y"]    = "e";
	m_pacPTS["expOffset"]	= "exposureparams";
}

QMap<QString, PTSVAL> PTSParser::parsePTS(QString filename)
{
	m_ptsConfig.clear();

    if (filename.isEmpty() || filename.trimmed() == "") {
        qDebug() << "File name is invalid!";
		return m_ptsConfig;
    }

    QFile ptsFile(filename);

    if (!ptsFile.exists()) {
        qDebug() << "\"" << filename << "\" is not exist!";
		return m_ptsConfig;
    }

    if (ptsFile.size() < 1024) {
        qDebug() << "File name is invalid! (Size is small)";
		return m_ptsConfig;
    }

    if (!ptsFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "\"" << filename << "\" is not text file!";
    }

	curBlock = ROOT_BLOCK;
	curImageIndex = 0;
	sharedValues = false;

    while (!ptsFile.atEnd()) {
        QString line = ptsFile.readLine();
		if (line == "# PTGui Trial Project File")
		{
			qDebug() << "\"" << filename << "\" is not text file!";
			return m_ptsConfig;
		}
        processLine(line);
    }

    int imgIndex = 0;
    while (m_ptsConfig.contains(QString("%1%2").arg(IMAGE_KEY).arg(QString::number(imgIndex)))) {
        PTSVAL entry = m_ptsConfig[QString("%1%2").arg(IMAGE_KEY).arg(QString::number(imgIndex))];
        //qDebug() << endl << entry.value;
        QMap<QString, float> params = entry.camParams;
        QMap<QString, float> shareParams = m_ptsConfig[SHARE_KEY].camParams;
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
			if (key == EXPOSURE_VAL && m_ptsConfig.contains(EXPOSURE_KEY + QString::number(imgIndex))) {
				QString expOffsetStr = m_ptsConfig[EXPOSURE_KEY + QString::number(imgIndex)].value.at(0);
				value = expOffsetStr.toFloat();
			}
            //qDebug() << QString("%1 (%2) = %3").arg(i.key()).arg(key).arg(value);
        }
        imgIndex++;
    }
    return m_ptsConfig;
}

int PTSParser::processLine(QString line)
{
    QString firstCh = line.left(2);
    QString lastCh = line.right(2);

    if (firstCh == COMMENT_EXP && lastCh != BLOCK_EXP) {    // Comment line
        return 0;
    }

    if (firstCh == COMMENT_EXP && lastCh == BLOCK_EXP) {    // Start Block
        QString block = line.mid(2, line.length()-4);
        //qDebug() << "Start Block: " << block;
        curBlock = block;
        return 0;
    }

    if (firstCh == VALUE_EXP) {                             // Value
        QString valStr = line.mid(2, line.length() - 3);
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

        if (valName == IMAGE_VAL) {
            valName += QString::number(curImageIndex++);
            sharedValues = false;
        }
        else if (valName == SHARE_VAL) {
            sharedValues = true;
        }
		else if (valName == EXPOSURE_VAL) {
			valName += QString::number(curImageIndex-1);
			sharedValues = false;
		}

        QString key = curBlock + "->" + valName;
        PTSVAL entry = {valType, tokens, tokens.length()};

		// #-exposureparams 0 0 0 1.097
		if (valName.startsWith(EXPOSURE_VAL)) {
			entry.type = NUMBER_VAL;
			entry.value.removeFirst();
			entry.value.removeFirst();
			entry.value.removeFirst();
			entry.nbVal = 1;
		}

        QString log = QString("key=%1, type=%2, value=%3").arg(key).arg(QString::number(valType)).arg(tokens.join(" "));
        //qDebug() << log;
        m_ptsConfig[key] = entry;
    }

    // ex: o f3 y86.61479670316621 r90.91172352473242 p-15.42569331521862 v=0 a=0 b=0 c=0 d=0 e=0 g=0 t=0
    if (firstCh == CAMERA_EXP) {
        QString valStr = line.mid(2, line.length() - 3);
        QStringList tokens = valStr.split(" ");
        QMap<QString, float> valEnties;
		QMap<QString, int> mapEntities;
        for (int i = 0; i < tokens.length(); i++) {
            QString exp = tokens.at(i);
            QString name = exp.left(1).toLower();
            QString val = "";
			bool isRef = false;
            if (exp.length() > 2 && exp.mid(1,1) == "=") {      // x=y
                val = exp.right(exp.length() - 2);
				isRef = true;
			}
			else // y86.61479670316621
			{
				val = exp.right(exp.length() - 1);
				isRef = false; // value
			}

            bool fflag = false;
            float fVal = val.toFloat(&fflag);
			if (fflag == true)
			{
				if (isRef)
					mapEntities[name] = fVal;
				else
					valEnties[name] = fVal;
			}
            //qDebug() << QString("%1=%2").arg(name).arg(QString::number(fVal));
        }
        QString key = curBlock + "->" + (sharedValues ? SHARE_VAL : IMAGE_VAL + QString::number(curImageIndex-1));
        PTSVAL entry = m_ptsConfig[key];
        if (valEnties.size() > 0)
            entry.camParams = valEnties;
		entry.camParamRefs = mapEntities;
        m_ptsConfig[key] = entry;
        //qDebug() << QString("key=%1, camParams=%2").arg(key).arg(valStr);
    }

    return 0;
}

