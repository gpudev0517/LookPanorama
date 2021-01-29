#ifndef PTSPARSER_H
#define PTSPARSER_H

#include <QString>
#include <QMap>

#include "pts.h"

class PTSParser
{
public:
    PTSParser();
    QMap<QString, PTSVAL> parsePTS(QString filename);

private:
    int processLine(QString line);

    QMap<QString, PTSVAL>   m_ptsConfig;
    QMap<QString, QString>  m_pacPTS;
};

#endif // PTSPARSER_H
