#include "Pixmap.h"
#include "ConfigZip.h"

Pixmap::Pixmap()
{
//	pix = NULL;
}

Pixmap::~Pixmap()
{
//	if (pix){
//		delete pix;
//		pix = NULL;
//	}

	for (int i = 0; i < pixList.size(); i++)
	{
		QPixmap *pixmap = pixList[i];
		if (pixmap) {
			delete pixmap;
			pixmap = NULL;
		}
	}

	pixList.clear();
}

void Pixmap::load(QString url) {

	ConfigZip cfz;
	QString rigiconPath = url.left(url.lastIndexOf("/")) + "/"+cfz.getValue(url,"D360","icon");
	if (cfz.getValue(url, "D360", "icon").isEmpty() || !cfz.extractZipFile(url, rigiconPath)) {
		pixList.append(NULL);
		return;
	}

//	QPixmap * old = 0;
//	if (pix) old = pix;
//	pix = new QPixmap(rigiconPath);

	QPixmap* pixmap = new QPixmap(rigiconPath);
	pixList.append(pixmap);

	emit dataChanged();

//	if (old) delete old;

	QFile::remove(rigiconPath);
}

void Pixmap::clear() {
//	if (pix) delete pix;
//	pix = 0;
//	emit dataChanged();
}

QString  Pixmap::data() {
	if (pixList[curIndex]) return "image://pixmap/"+QString::number((qulonglong)pixList[curIndex]);
	else return QString("../resources/icon_tempCamera.png");
}

void Pixmap::setCurIndex(int index)
{
	curIndex = index;
}