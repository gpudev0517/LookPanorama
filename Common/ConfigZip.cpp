#include "ConfigZip.h"
#include <QFile>
#include <QSettings>
#include "define.h"


ConfigZip::ConfigZip()
{
}

ConfigZip::~ConfigZip()
{
}

bool ConfigZip::addToZip(zip *za, QString & src, QString & src_fileName, QString & dst_dir, zip_int32_t cm_mode /* = ZIP_CM_DEFLATE */)
{
	long len = src.toUtf8().length();
	char *teststr = (char*)malloc(len + 1);
	memset(teststr, 0, len + 1);
	strncpy(teststr, src.toUtf8(), len);
	bool writeSucceed = true;
	zip_source *zs = NULL;
	if ((zs = zip_source_buffer(za, teststr, len, 0)) == NULL) {
		zip_strerror(za);
		writeSucceed = false;
	}
	if (dst_dir == "")
		dst_dir = src_fileName;
	else
		dst_dir.append("/").append(src_fileName);
	zip_int64_t index = zip_add(za, dst_dir.toUtf8().data(), zs);
	zip_set_file_compression(za, index, cm_mode, cm_mode);
	if (index == -1) {
		zip_source_free(zs);
	}
	return writeSucceed;
}

bool ConfigZip::addFileToZip(zip *za, QString & src_path, QString & src_fileName, QString & dst_dir, bool isDelete, zip_int32_t cm_mode /* = ZIP_CM_DEFLATE */)
{
	bool writeSucceed = true;
	if (!QFile::exists(src_path))
		return false;
	zip_source *zs = NULL;
	if ((zs = zip_source_file(za, src_path.toUtf8().data(), 0, -1)) == NULL) {
		fprintf(stderr, "error creating file source for '%s': %s\n", (const char*)src_path.data(), zip_strerror(za));
		writeSucceed = false;
	}
	if (dst_dir == "")
		dst_dir = src_fileName;
	else
		dst_dir.append("/").append(src_fileName);
	zip_int64_t index = zip_add(za, dst_dir.toUtf8().data(), zs);
	zip_set_file_compression(za, index, cm_mode, cm_mode);
	if (index == -1) {
		zip_source_free(zs);
	}
	if (isDelete)
		QFile::remove(src_path);
	return writeSucceed;
}

bool ConfigZip::createZipFile(QString & zip_fileName, QString add_fileName)
{
	zip* za = NULL;
	int err;
	int openType = ZIP_CREATE;

	za = zip_open((const char *)zip_fileName.toUtf8().data(), openType, &err);
	if (!za)
	{
		return false;
	}
	bool zipSucces = false;
	if (QFile::exists(add_fileName))
	{
		QString fileName = add_fileName.mid(add_fileName.lastIndexOf("/"));
		zipSucces = addFileToZip(za, add_fileName, add_fileName.mid(add_fileName.lastIndexOf("/")+1), QString(""), false, ZIP_CM_STORE);
	}
	zip_close(za);
	if (zipSucces)
		QFile::remove(add_fileName);
	return zipSucces;
}

bool ConfigZip::extractZipFile(QString & zip_fileName, QString extract_fileName)
{
	int err;
	zip *za = NULL;
	zip_file *zf = NULL;

	QString destFilePath = zip_fileName;
	QString fileName;
	//destFilePath.replace("/", "\\");
	if (!QFile::exists(destFilePath))
		return false;

	QByteArray rawContent;
	za = zip_open((const char *)destFilePath.toUtf8(), 0, &err);
	if (!za)
	{
		return false;
	}

	bool bSucces = extractFileToZip(za, zf, zip_fileName, extract_fileName.mid(extract_fileName.lastIndexOf("/")+1), QString(""), rawContent);
	zip_close(za);
	if (!bSucces)
		return false;
	QFile contectFile(extract_fileName);
	if (!contectFile.open(QIODevice::WriteOnly))
		return false;
	contectFile.write(rawContent);
	rawContent.clear();
	contectFile.close();
	return true;
}

bool ConfigZip::extractFileToZip(zip* za, zip_file *zf, QString & zip_fileName, QString & src_fileName, QString & dst_dir, QByteArray & dataArray)
{
	int err, n;
	char b[8192];
	memset(b, 0, sizeof(b));
	if (dst_dir == "")
		dst_dir = src_fileName;
	else
		dst_dir.append("/").append(src_fileName);

	if ((zf = zip_fopen(za, dst_dir.toUtf8(), 0)) != NULL) {
		while ((n = zip_fread(zf, b, sizeof(b)-2)) > 0)
		{
			dataArray.append(b, n);
			memset(b, 0, sizeof(b));
		}
		err = zip_fclose(zf);
		return true;
	}
	return false;
}

QString ConfigZip::getValue(QString l3dPath, QString groupStr, QString childKey)
{
	QString valueString = "";
	QString iniFileName = l3dPath.left(l3dPath.lastIndexOf("/")) + "/configuration.ini";
	bool extraSucces = extractZipFile(l3dPath, iniFileName);
	if (!extraSucces)
		return "";

	QSettings settings(iniFileName, QSettings::IniFormat);
	QFile::remove(iniFileName);
	settings.beginGroup(groupStr);
	const QStringList Keys = settings.childKeys();
	foreach(const QString &Key, Keys)
	{
		if (Key == childKey)
		{
			valueString = settings.value(Key).toString();
		}
	}
	settings.endGroup();
	return valueString;
}