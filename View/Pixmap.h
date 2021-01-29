#ifndef _PIXMAP_H
#define _PIXMAP_H

#include <QObject>
#include <QPixmap>
#include <QQuickImageProvider>
#include <QQuickItem>

class Pixmap : public QObject {
	Q_OBJECT
	Q_PROPERTY(QString data READ data NOTIFY dataChanged)
public:
	Pixmap();
	virtual ~Pixmap();

	QString data();

	public slots:
	void load(QString url);
	void clear();
	void setCurIndex(int i);

signals:
	void dataChanged();

private:
//	QPixmap * pix;
	QList<QPixmap*> pixList;
	int curIndex;
};


class PixmapProvider : public QQuickImageProvider {
public:
	PixmapProvider() : QQuickImageProvider(QQuickImageProvider::Pixmap) {}
	QPixmap requestPixmap(const QString &id, QSize *size, const QSize &requestedSize) {
		qulonglong d = id.toULongLong();
		if (d) {
			QPixmap * p = reinterpret_cast<QPixmap *>(d);
			return *p;
		}
		else {
			return QPixmap();
		}
	}
};

#endif // !_PIXMAP_H
