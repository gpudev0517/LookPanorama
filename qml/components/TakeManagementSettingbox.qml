import QtQuick 2.5
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls 1.4
import QtQml.Models 2.2
import QtQuick.Dialogs 1.2
import "../controls"

ToolWindow{
    id: details
    width: 300
    height: 500
    z: 10
    windowState: qsTr("windowed")
    visible: false

    property string   title: "Take Management"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20

    Item {
        id: takeSaveItem
        width: parent.width
        height: 30
        anchors.top: parent.top
        anchors.topMargin: 20
        Text {
            id: takeSaveText
            y: 12
            color: "#ffffff"
            text: qsTr("Capture Path")
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14

        }

        FlatText {
            id: takeSavePath
            width: parent.width * 0.4
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: takeSaveText.right
            anchors.leftMargin: spacing
            text:  (qmlMainWindow.applicationSetting.sessionTkgCapturePath == "")? "C:/Capture": qmlMainWindow.applicationSetting.sessionTkgCapturePath

			onAccepted: {
				console.log("New capture path: ", text)
				qmlMainWindow.setSessionRootPath(text);
			}

			onTextChanged: {
                qmlMainWindow.applicationSetting.setSessionTkgCapturePath(text)
        	}
        }

        Rectangle{
            width: 30
            height: 30
            z: 1
            anchors.right: parent.right
            anchors.rightMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            color: "#373737"
            //border.color: "#4e8d15"
            Text{
                anchors.fill: parent
                text: "  ..."
                color: "#4e8d15"
                verticalAlignment: Text.AlignVCenter
                z: 3
            }

        }
        Rectangle{
            id: saveHoveredRectangle
            width: 30
            height: 30
            z: 1
            color: "#373737"
            anchors.right: parent.right
            anchors.rightMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            border.color: "#4e8d15"
            border.width: 1
            Text{
                anchors.fill: parent
                text: "  ..."
                color: "#4e8d15"
                verticalAlignment: Text.AlignVCenter
                z: 3
            }

            visible: false
        }
        MouseArea{
            x: saveHoveredRectangle.x
            z: 2
            width: saveHoveredRectangle.width
            height: saveHoveredRectangle.height
            anchors.verticalCenter: parent.verticalCenter
            hoverEnabled: true
            onEntered: saveHoveredRectangle.visible = true
            onExited: saveHoveredRectangle.visible = false
            onClicked: saveFileDialoge.visible = true
        }


        FileDialog{
            id: saveFileDialoge
            title: "Select capture path"
            selectMultiple: false
            selectFolder: true;

            onSelectionAccepted: {
                var fileName = fileUrl.toString().substring(8); // Remove "file:///" prefix
                takeSavePath.text = fileName;
                qmlMainWindow.setSessionRootPath(takeSavePath.text);
            }
        }
    }

	Item {
        id: takeCommentItem
        width: parent.width
        height: 30
        anchors.top: takeSaveItem.bottom
        anchors.topMargin: 20
        Text {
            id: lblTakeComment
            y: 12
            color: "#ffffff"
            text: qsTr("Comment     ")
            anchors.left: parent.left
            anchors.leftMargin: spacing
            anchors.verticalCenter: parent.verticalCenter
            font.pixelSize: 14
        }

        FlatText {
            id: txtTakeComment
            width: parent.width * 0.4;
            height: 30
            anchors.verticalCenter: parent.verticalCenter
            anchors.left: lblTakeComment.right
            anchors.leftMargin: spacing
			anchors.rightMargin: spacing

			onAccepted: {
				
			}
		}

		Item {
        id: saveTakeItem
        width:saveHoveredRectangle.width;
        height: 30
        anchors.right: parent.right
        anchors.rightMargin: spacing
		anchors.verticalCenter: parent.verticalCenter

        Rectangle {
            id: saveTakeHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#373737"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: 1
        }


        Text {
            color: "#ffffff"
            text: "Save"
            z: 1
            font.pointSize: 10
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent


        }

        Rectangle {
            id: saveTakeComment
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#373737"

            MouseArea {
                id: saveMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        saveTakeHoverRect.visible = true;
                    }else{
                        saveTakeHoverRect.visible = false;
                    }
                }

                onClicked: {
                    qmlMainWindow.changeComment(takeTreeView.currentIndex, txtTakeComment.text);
                }
            }
        }
	}
    }

    property string backColor: "#262626"

    TreeView {
		id: takeTreeView
        //anchors.fill: parent
        //anchors.margins : spacing * 2
        anchors.top: takeCommentItem.bottom
        anchors.topMargin: spacing
        anchors.left: parent.left
        anchors.leftMargin: spacing
        width: parent.width - spacing * 2
        height: parent.height - spacing * 9
        TableViewColumn {
            title: "Session and Take"
            role: "display"
            width: 300 - spacing * 2.25
        }

        selection: ItemSelectionModel {
		id: takeSelModel
        	model: takeMgrModel
        	onSelectionChanged: {
            }
        }

		model: takeMgrModel

        onClicked: {
			txtTakeComment.text = qmlMainWindow.getTakeComment(index);
        }

        onDoubleClicked: {
            var isTake = qmlMainWindow.isTakeNode(index);
            if (isTake) {
				if(root.panoMode == 1) {
					root.onChangePanoMode();
					root.initPlayback(index);
				}
            } else {
                isExpanded(index) ? collapse(index) : expand(index)
            }
        }


        Component.onCompleted: {
			qmlMainWindow.setTreeView(this); 
        }


        style: TreeViewStyle{
               backgroundColor : backColor
               alternateBackgroundColor: backColor
               /*branchDelegate: Item {
                   width: 16
                   height: 16
                   Text {
                       visible: styleData.column === 0 && styleData.hasChildren
                       text: styleData.isExpanded ? "\u25bc" : "\u25b6"
                       color: !control.activeFocus || styleData.selected ?  "#DDDDDD" : "#DDDDDD"
                       font.pointSize: 10
                       anchors.centerIn: parent
                       anchors.verticalCenterOffset: styleData.isExpanded ? 2 : 0
                   }
               }*/
               itemDelegate: Item {
                   height: Math.max(20, label.implicitHeight)
                   property int implicitWidth: label.implicitWidth + 20

                   Text {
                       id: label
                       objectName: "label"
                       width: parent.width - x
                       x: styleData.depth && styleData.column === 0 ? 0 : 8
                       horizontalAlignment: styleData.textAlignment
                       anchors.verticalCenter: parent.verticalCenter
                       anchors.verticalCenterOffset: 1
                       elide: styleData.elideMode
                       text: styleData.value !== undefined ? styleData.value : ""
                       font.pixelSize: 10
                       color: "#ffffff"
                   }
               }
               headerDelegate: Rectangle {
                   height: textItem.implicitHeight * 1.3
                   color: "#373737"
                   anchors.topMargin: 2
                   Text {
                       id: textItem
                       anchors.fill: parent
                       verticalAlignment: Text.AlignVCenter
                       horizontalAlignment: styleData.textAlignment
                       anchors.leftMargin: 12
                       text: styleData.value
                       elide: Text.ElideRight
                       color:  "#ffffff"
                       font.pixelSize: 12
                   }
                   Rectangle {
                       height: 1
                       border.color: backColor
                       color:  backColor
                       anchors.bottom: parent.bottom
                       anchors.left: parent.left
                       anchors.right: parent.right
                   }
                   Rectangle {
                       height: 1
                       border.color: backColor
                       color:  backColor
                       anchors.top: parent.top
                       anchors.left: parent.left
                       anchors.right: parent.right
                   }
                   Rectangle {
                       width: 1
                       border.color: backColor
                       color: backColor
                       anchors.bottom: parent.bottom
                       anchors.top: parent.top
                       anchors.right: parent.right
                   }
               }
               frame: Rectangle {
                   color: backColor
                   border.color: backColor
                   border.width: 1
               }
               handle: Rectangle {
                   color: "#505050"
                   implicitHeight: 4
                   implicitWidth:  4
                   radius: 3
               }

               scrollBarBackground: Item {
                   implicitWidth: 0
               }
               decrementControl: Rectangle{
                   visible: false
               }

               incrementControl: Rectangle{
                   visible: false
               }
               corner: Rectangle{
                   visible:false
               }
           }
	}

    Item {
        id: newSesionItem
        width: 80
        anchors.bottom: parent.bottom
		anchors.bottomMargin: 10
        height: 30
        anchors.topMargin: 10
        anchors.right: parent.right
        anchors.rightMargin: spacing

        Rectangle {
            id: newSessionHoverRect
            x: 0
            width: parent.width
            height: parent.height
            anchors.fill: parent
            color: "#373737"
            visible: false
            border.color: "#4e8d15"
            border.width: 1
            z: 1
        }


        Text {
            color: "#ffffff"
            text: "New Session"
            z: 1
            font.pointSize: 10
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
            anchors.fill: parent


        }

        Rectangle {
            id: newSessionRect
            width: parent.width
            height: parent.height
           anchors.fill: parent
            color: "#373737"

            MouseArea {
                id: detailMouseArea
                width: 60
                anchors.fill: parent
                hoverEnabled: true
                onHoveredChanged: {
                    isHovered = !isHovered
                    if(isHovered){
                        newSessionHoverRect.visible = true;
                    }else{
                        newSessionHoverRect.visible = false;
                    }
                }

                onClicked: {
                    qmlMainWindow.createNewSession();
                    statusItem.updateTakeMgr();
                }
            }
        }
    }

    function initTakeManagement() {
		takeSavePath.text = qmlMainWindow.applicationSetting.sessionTkgCapturePath
		txtTakeComment.text = qsTr("");
    }

	function getTakeManangementComment() {
		return txtTakeComment.text;
	}

	function setCurrentModelIndex(index) {
		takeSelModel.clearSelection();
		takeSelModel.setCurrentIndex(index, 0x0002);
        takeSelModel.select(index, 0x0002);
    }
}
