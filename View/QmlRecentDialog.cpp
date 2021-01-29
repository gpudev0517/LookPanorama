#include "QmlRecentDialog.h"
#include <qfile.h>

QmlRecentDialog::QmlRecentDialog()
{
	m_imageID = 0;
	m_title = "";
	m_sourcePath = "";
}


QmlRecentDialog::~QmlRecentDialog()
{
	
}

QString QmlRecentDialog::title() const
{
	return m_title;
}
QString QmlRecentDialog::sourcePath() const
{
	return m_sourcePath;
}
int QmlRecentDialog::imageID() const
{
	return m_imageID;
}
void QmlRecentDialog::setTitle(QString title)
{
	m_title = title;
	emit titleChanged(m_title);
}
void QmlRecentDialog::setSourcePath(QString sourcepath)
{
	m_sourcePath = sourcepath;
	emit sourcePathChanged(m_sourcePath);
}
void QmlRecentDialog::setImageID(int id)
{
	m_imageID = id;
	emit imageIDChanged(m_imageID);
}
