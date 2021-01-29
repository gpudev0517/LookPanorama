#include "QmlTakeManagement.h"
#include <qdebug.h>
#include <QtCore>
#include <QVariant>

QmlTakeManagement::QmlTakeManagement(QObject* parent) : QObject(parent)
{
	m_lastSessionName = " ";
	m_takeName = "";
	m_takeDir = "";
	m_currentTakeList.clear();
	
	QList<QModelIndex> m_ModelList;
}

QmlTakeManagement::~QmlTakeManagement()
{
}

bool QmlTakeManagement::isTakeNode(TakeMgrTreeModel* model, QModelIndex index)
{

	QModelIndex parentInd = index.parent();
	if (parentInd.isValid())
		return true;
	return false;
}

QModelIndex QmlTakeManagement::prevModelIndex(QModelIndex index)
{
	for (int i = 0; i < m_ModelList.size(); i++)
	{
		if (index == m_ModelList.at(i))
		{
			int prev = i - 1;
			if (prev >= 0)
				return m_ModelList.at(prev);
			else
				return QModelIndex();
		}
	}

	return QModelIndex();
}

QModelIndex QmlTakeManagement::nextModelIndex(QModelIndex index)
{
	for (int i = 0; i < m_ModelList.size(); i++)
	{
		if (index == m_ModelList.at(i))
		{
			int next = i + 1;
			if (next < m_ModelList.size())
				return m_ModelList.at(next);
			else
				return QModelIndex();
		}
	}

	return QModelIndex();
}

void QmlTakeManagement::insertSession(TakeMgrTreeModel* model, QString data, int sessionid)
{
	TreeItem *parent = model->rootItem();
	QModelIndex index = model->index(0, 0);

	if (!model->insertRow(sessionid, index.parent()))
		return;

	QModelIndex child = model->index(sessionid, 0, index.parent());
	model->setData(child, QVariant(data), Qt::EditRole);
}

QModelIndex QmlTakeManagement::insertTake(TakeMgrTreeModel* model, QString data, int sessionid, int takeid)
{
	QModelIndex index = model->index(sessionid, 0);	
	if (!model->insertRow(takeid, index))
		return QModelIndex();
	QModelIndex child = model->index(takeid, 0, index);
	model->setData(child, QVariant(data), Qt::EditRole);
	m_ModelList.append(child);
	return child;
}

void QmlTakeManagement::removeAllSession(TakeMgrTreeModel* model, int position, int sessionCount)
{
	model->removeRows(position, sessionCount);
}

void QmlTakeManagement::loadTree(TakeMgrTreeModel* model,QString capturePath)
{
	if (m_GlobalSessionNameList.size() != 0){
		model->removeRows(0, m_GlobalSessionNameList.size());
	}
	
	setupModelData(capturePath);
	
	int sessionId = 0;
	int takeId = 0;
	QString takeName = "";
	std::vector<QString> m_currentTakeList; 

	if (!m_GlobalSessionNameList.size())
		return;
	for (int i = 0; i < m_GlobalSessionNameList.size(); i++)
	{
		QString str = m_GlobalSessionNameList[i];
		m_lastSessionName = str;
		insertSession(model, str, sessionId);
		sessionId++;
		takeId = 0;

		m_currentTakeList.clear();
		m_currentTakeList = m_GlobalTakeNameList[i];

		if (m_currentTakeList.size() > 0)
		{
			for (int j = 0; j < m_currentTakeList.size(); j++)
			{
				m_takeName = m_currentTakeList[j];
				insertTake(model, m_takeName, sessionId, takeId);
				takeId++;

			}
		}
	}

	g_globalSessionId = m_lastSessionName.right(2).toInt();
}
