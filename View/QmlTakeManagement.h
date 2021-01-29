#ifndef QMLTAKEMANAGEMENT_H
#define QMLTAKEMANAGEMENT_H

#include <QObject>
#include <QStringList>
#include "TakeMgrTreeModel.h"

class QmlTakeManagement : public QObject
{
	Q_OBJECT

public:
	QmlTakeManagement(QObject* parent = 0);
	virtual ~QmlTakeManagement();

	bool isTakeNode(TakeMgrTreeModel* model, QModelIndex index);
	void insertSession(TakeMgrTreeModel* model, QString data, int sessionid);
	QModelIndex insertTake(TakeMgrTreeModel* model, QString data, int sessionid, int takeid);
	void removeAllSession(TakeMgrTreeModel* model, int position, int sessionCount);
	QModelIndex prevModelIndex(QModelIndex index);
	QModelIndex nextModelIndex(QModelIndex index);

public slots:
	void loadTree(TakeMgrTreeModel* model, QString capturePath);
private:
	QString m_lastSessionName;
	QString m_takeName;
	QString m_takeDir;
	std::vector<QString> m_currentTakeList;
	QList<QModelIndex> m_ModelList;
};

#endif // QMLTAKEMANAGEMENT_H
