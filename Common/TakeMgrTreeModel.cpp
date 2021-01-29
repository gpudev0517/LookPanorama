#include "TakeMgrTreeModel.h"
#include <QDir>

std::vector<QString> m_GlobalSessionNameList = {};
std::vector<std::vector<QString>> m_GlobalTakeNameList = {};
int				g_globalSessionId = 0;

TreeItem::TreeItem(const QVector<QVariant> &data, TreeItem *parent)
{
	m_parentItem = parent;
	m_itemData = data;
}
//! [0]

//! [1]
TreeItem::~TreeItem()
{
	qDeleteAll(m_childItems);
}
//! [1]

//! [2]
TreeItem *TreeItem::child(int number)
{
	return m_childItems.value(number);
}
//! [2]

//! [3]
int TreeItem::childCount() const
{
	return m_childItems.count();
}
//! [3]

//! [4]
int TreeItem::childNumber() const
{
	if (m_parentItem)
		return m_parentItem->m_childItems.indexOf(const_cast<TreeItem*>(this));

	return 0;
}
//! [4]

//! [5]
int TreeItem::columnCount() const
{
	return m_itemData.count();
}
//! [5]

//! [6]
QVariant TreeItem::data(int column) const
{
	return m_itemData.value(column);
}
//! [6]

//! [7]
bool TreeItem::insertChildren(int position, int count, int columns)
{
	if (position < 0 || position > m_childItems.size())
		return false;

	for (int row = 0; row < count; ++row) {
		QVector<QVariant> data(columns);
		TreeItem *item = new TreeItem(data, this);
		m_childItems.insert(position, item);
	}

	return true;
}
//! [7]

//! [8]
bool TreeItem::insertColumns(int position, int columns)
{
	if (position < 0 || position > m_itemData.size())
		return false;

	for (int column = 0; column < columns; ++column)
		m_itemData.insert(position, QVariant());

	foreach(TreeItem *child, m_childItems)
		child->insertColumns(position, columns);

	return true;
}
//! [8]

//! [9]
TreeItem *TreeItem::parent()
{
	return m_parentItem;
}
//! [9]

//! [10]
bool TreeItem::removeChildren(int position, int count)
{
	if (position < 0 || position + count > m_childItems.size())
		return false;

	for (int row = 0; row < count; ++row)
		delete m_childItems.takeAt(position);

	return true;
}
//! [10]

bool TreeItem::removeColumns(int position, int columns)
{
	if (position < 0 || position + columns > m_itemData.size())
		return false;

	for (int column = 0; column < columns; ++column)
		m_itemData.remove(position);

	foreach(TreeItem *child, m_childItems)
		child->removeColumns(position, columns);

	return true;
}

//! [11]
bool TreeItem::setData(int column, const QVariant &value)
{
	if (column < 0 || column >= m_itemData.size())
		return false;

	m_itemData[column] = value;
	return true;
}

TakeMgrTreeModel::TakeMgrTreeModel(const QStringList &headers, const QString &data, QObject *parent)
: QAbstractItemModel(parent)
{
	QVector<QVariant> rootData;
	foreach(QString header, headers)
		rootData << header;

	m_rootItem = new TreeItem(rootData);
	//setupModelData();
}
//! [0]

//! [1]
TakeMgrTreeModel::~TakeMgrTreeModel()
{
	delete m_rootItem;
}
//! [1]

//! [2]
int TakeMgrTreeModel::columnCount(const QModelIndex & /* parent */) const
{
	return m_rootItem->columnCount();
}
//! [2]

QVariant TakeMgrTreeModel::data(const QModelIndex &index, int role) const
{
	if (!index.isValid())
		return QVariant();

	if (role != Qt::DisplayRole && role != Qt::EditRole)
		return QVariant();

	TreeItem *item = getItem(index);

	return item->data(index.column());
}

//! [3]
Qt::ItemFlags TakeMgrTreeModel::flags(const QModelIndex &index) const
{
	if (!index.isValid())
		return 0;

	return Qt::ItemIsEditable | QAbstractItemModel::flags(index);
}
//! [3]

//! [4]
TreeItem *TakeMgrTreeModel::getItem(const QModelIndex &index) const
{
	if (index.isValid()) {
		TreeItem *item = static_cast<TreeItem*>(index.internalPointer());
		if (item)
			return item;
	}
	return m_rootItem;
}
//! [4]

QVariant TakeMgrTreeModel::headerData(int section, Qt::Orientation orientation,
	int role) const
{
	if (orientation == Qt::Horizontal && role == Qt::DisplayRole)
		return m_rootItem->data(section);

	return QVariant();
}

//! [5]
QModelIndex TakeMgrTreeModel::index(int row, int column, const QModelIndex &parent) const
{
	if (parent.isValid() && parent.column() != 0)
		return QModelIndex();
	//! [5]

	//! [6]
	TreeItem *m_parentItem = getItem(parent);

	TreeItem *childItem = m_parentItem->child(row);
	if (childItem)
		return createIndex(row, column, childItem);
	else
		return QModelIndex();
}
//! [6]

bool TakeMgrTreeModel::insertColumns(int position, int columns, const QModelIndex &parent)
{
	bool success;

	beginInsertColumns(parent, position, position + columns - 1);
	success = m_rootItem->insertColumns(position, columns);
	endInsertColumns();

	return success;
}

bool TakeMgrTreeModel::insertRows(int position, int rows, const QModelIndex &parent)
{
	TreeItem *m_parentItem = getItem(parent);
	bool success;

	beginInsertRows(parent, position, position + rows - 1);
	success = m_parentItem->insertChildren(position, rows, m_rootItem->columnCount());
	endInsertRows();

	return success;
}

//! [7]
QModelIndex TakeMgrTreeModel::parent(const QModelIndex &index) const
{
	if (!index.isValid())
		return QModelIndex();

	TreeItem *childItem = getItem(index);
	TreeItem *m_parentItem = childItem->parent();

	if (m_parentItem == m_rootItem)
		return QModelIndex();

	return createIndex(m_parentItem->childNumber(), 0, m_parentItem);
}
//! [7]

bool TakeMgrTreeModel::removeColumns(int position, int columns, const QModelIndex &parent)
{
	bool success;

	beginRemoveColumns(parent, position, position + columns - 1);
	success = m_rootItem->removeColumns(position, columns);
	endRemoveColumns();

	if (m_rootItem->columnCount() == 0)
		removeRows(0, rowCount());

	return success;
}

bool TakeMgrTreeModel::removeRows(int position, int rows, const QModelIndex &parent)
{
	TreeItem *m_parentItem = getItem(parent);
	bool success = true;

	beginRemoveRows(parent, position, position + rows - 1);
	success = m_parentItem->removeChildren(position, rows);
	endRemoveRows();

	return success;
}

//! [8]
int TakeMgrTreeModel::rowCount(const QModelIndex &parent) const
{
	TreeItem *m_parentItem = getItem(parent);

	return m_parentItem->childCount();
}
//! [8]

bool TakeMgrTreeModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	if (role != Qt::EditRole)
		return false;

	TreeItem *item = getItem(index);
	bool result = item->setData(index.column(), value);

	if (result)
		emit dataChanged(index, index);

	return result;
}

bool TakeMgrTreeModel::setHeaderData(int section, Qt::Orientation orientation,
	const QVariant &value, int role)
{
	if (role != Qt::EditRole || orientation != Qt::Horizontal)
		return false;

	bool result = m_rootItem->setData(section, value);

	if (result)
		emit headerDataChanged(orientation, section, section);

	return result;
}

void setupModelData(QString capturePath)
{
	m_GlobalSessionNameList.clear();
	m_GlobalTakeNameList.clear();
	QString strTravelResult = "";
	QString sessionRootPath = capturePath;
	std::vector<QString> currentTakeList;


	// Navigate all sub directories in sessionRootDirPath	
	QDir rootPath(sessionRootPath);
	QStringList dirList = rootPath.entryList(QDir::Dirs);
	foreach(const QString dirItem, dirList) {
		if (dirItem == "." || dirItem == "..")
			continue;
		m_GlobalSessionNameList.push_back(dirItem);
		int sessionId = dirItem.right(2).toInt();
		g_globalSessionId = sessionId;
		currentTakeList.clear();
		m_GlobalTakeNameList.push_back(currentTakeList);
		
		QString strSessionPath = QString("%1/%2").arg(sessionRootPath).arg(dirItem);
		QDir sessionPath(strSessionPath);
		QStringList fileList = sessionPath.entryList(QDir::Files);
		foreach(const QString fileItem, fileList) {
			if (fileItem.startsWith("Take_") && fileItem.endsWith(".mp4"))
			{
				currentTakeList.push_back(fileItem);
			}
			
		}

		m_GlobalTakeNameList[g_globalSessionId] = currentTakeList;
	}
}