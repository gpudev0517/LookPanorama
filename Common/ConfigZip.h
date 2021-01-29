#ifndef CONFIGZIP_H
#define CONFIGZIP_H

#include <QObject>
#include <QString>
#include "zip.h"

class ConfigZip: public QObject
{
	Q_OBJECT

public:
	ConfigZip();
	~ConfigZip();

	bool addToZip(zip *za, QString & src, QString & src_fileName, QString & dst_dir, zip_int32_t cm_mode = ZIP_CM_DEFLATE);
	bool addFileToZip(zip *za, QString & src_path, QString & src_fileName, QString & dst_dir, bool isDelete, zip_int32_t cm_mode = ZIP_CM_DEFLATE);
	bool createZipFile(QString & zip_fileName, QString add_fileName);
	bool extractZipFile(QString & zip_fileName, QString extract_fileName);
	bool extractFileToZip(zip* za, zip_file *zf, QString & zip_fileName, QString & src_fileName, QString & dst_dir, QByteArray & dataArray);
	QString getValue(QString l3dPath, QString groupStr, QString childKey);

private:

};

#endif //CONFIGZIP_H