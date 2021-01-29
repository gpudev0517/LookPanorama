#ifndef TAKEMGRTREEMODEL_H
#define TAKEMGRTREEMODEL_H

#pragma once
#include <QAbstractItemModel>
#include <QModelIndex>
#include <QItemSelectionModel>

class TreeItem
{
public:
	explicit TreeItem(const QVector<QVariant> &data, TreeItem *parent = 0);
	virtual ~TreeItem();

	TreeItem *child(int number);
	int childCount() const;
	int columnCount() const;
	QVariant data(int column) const;
	bool insertChildren(int position, int count, int columns);
	bool insertColumns(int position, int columns);
	TreeItem *parent();
	bool removeChildren(int position, int count);
	bool removeColumns(int position, int columns);
	int childNumber() const;
	bool setData(int column, const QVariant &value);

private:
	QList<TreeItem*> m_childItems;
	QVector<QVariant> m_itemData;
	TreeItem * m_parentItem;
};


class TakeMgrTreeModel : public QAbstractItemModel
{
	Q_OBJECT

public:
	TakeMgrTreeModel(const QStringList &headers, const QString &data,
		QObject *parent = 0);
	virtual ~TakeMgrTreeModel();
	//! [0] //! [1]

	QVariant data(const QModelIndex &index, int role) const Q_DECL_OVERRIDE;
	QVariant headerData(int section, Qt::Orientation orientation,
		int role = Qt::DisplayRole) const Q_DECL_OVERRIDE;

	QModelIndex index(int row, int column,
		const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
	QModelIndex parent(const QModelIndex &index) const Q_DECL_OVERRIDE;

	int rowCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
	int columnCount(const QModelIndex &parent = QModelIndex()) const Q_DECL_OVERRIDE;
	//! [1]

	//! [2]
	Qt::ItemFlags flags(const QModelIndex &index) const Q_DECL_OVERRIDE;
	bool setData(const QModelIndex &index, const QVariant &value,
		int role = Qt::EditRole) Q_DECL_OVERRIDE;
	bool setHeaderData(int section, Qt::Orientation orientation,
		const QVariant &value, int role = Qt::EditRole) Q_DECL_OVERRIDE;

	bool insertColumns(int position, int columns,
		const QModelIndex &parent = QModelIndex()) Q_DECL_OVERRIDE;
	bool removeColumns(int position, int columns,
		const QModelIndex &parent = QModelIndex()) Q_DECL_OVERRIDE;
	bool insertRows(int position, int rows,
		const QModelIndex &parent = QModelIndex()) Q_DECL_OVERRIDE;
	bool removeRows(int position, int rows,
		const QModelIndex &parent = QModelIndex()) Q_DECL_OVERRIDE;
	TreeItem* rootItem() { return m_rootItem; }

private:
	TreeItem *getItem(const QModelIndex &index) const;

	TreeItem *m_rootItem;
};

extern std::vector<QString> m_GlobalSessionNameList;
extern std::vector<std::vector<QString>> m_GlobalTakeNameList;
extern int g_globalSessionId;
void setupModelData(QString capturePath);
#endif