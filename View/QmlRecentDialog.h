#pragma once
#include <QObject>
#include "D360Parser.h"

class QmlRecentDialog : public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString title READ title() NOTIFY titleChanged)
	Q_PROPERTY(QString sourcePath READ sourcePath NOTIFY sourcePathChanged)
	Q_PROPERTY(int imageID READ imageID NOTIFY imageIDChanged)

	public:
		QmlRecentDialog();
		virtual ~QmlRecentDialog();

		QString title() const;
		QString sourcePath() const;
		int imageID() const;

	private:
		QString m_title;
		QString m_sourcePath;
		int m_imageID;

	signals:
		void titleChanged(QString title);
		void sourcePathChanged(QString sourcePath);
		void imageIDChanged(int imageID);	

	public slots:
		void setTitle(QString title);
		void setSourcePath(QString sourcePath);
		void setImageID(int id);
};

