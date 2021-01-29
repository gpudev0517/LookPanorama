import QtQuick 2.0

Item {
    width: 350
    height: 50

    property string     typeText: "Error"
    property string     contentText: "Terminating audio and camera capture threads..."
    property string     imagePath: "../../resources/ico_error.png"
    property string     timeText
    property bool       isHovered: false
    property bool       isDeleteHovered: false

//    Rectangle {
//        id: background
//        width: 350
//        height: parent.height
//        color: "#171717"

//    }

    Rectangle {
        id: hoveredRectangle
//        y: -1
        width: parent.width
        height: parent.height
        color: "#353535"
        visible: false
    }

    Item {
        id: notificationItem
        anchors.top: parent.top
        anchors.topMargin: 0
        width: parent.width * 5 /7
        height:　parent.height

        Item {
            id: typeItem
            width: 50
            height: 50
            y: (parent.height - height) / 2

            Image {
                anchors.centerIn: parent
                y: (parent.height - height) / 2
                width: 40
                height: 40
                source: imagePath
                fillMode: Image.PreserveAspectFit
            }
        }

//        Item {
//            id: titleItem
//            width: parent.width - typeItem.width
//            height: parent.height / 3
//            anchors.left: typeItem.right

//            Text {
//                id: title
//                width: 100
//                height: 23
//                color: "#ffffff"
//                text: typeText
//                font.pointSize: 12
//                font.bold: true
//                verticalAlignment: Text.AlignVCenter
//                horizontalAlignment: Text.AlignLeft
//            }
//        }

        Item {
            id: contentItem
            y: (parent.height - height) / 2
            width: parent.width - typeItem.width
            height: parent.height * 2 / 3
            anchors.left: typeItem.right
            anchors.leftMargin: 10
            //anchors.top: titleItem.bottom

            Text {
                id: content
                y: (parent.height - height) / 2
                width: parent.width
                color: "#929292"
                text: contentText
                wrapMode: Text.WordWrap
                textFormat: Text.RichText
                font.pointSize: 12
//                anchors.bottom: parent.bottom
//                anchors.bottomMargin: 7
            }
        }
    }

    Spliter {
        id: horizontalSpliter
        width: parent.width
        anchors.bottom: parent.bottom
    }

    Item {
        id: verticalSpliter
        y: (parent.height - height) / 2
        width: 2
        height: 30
        anchors.left: notificationItem.right
        anchors.leftMargin: 30
        Rectangle{
            color: "#1f1f1f"
            x: 1
            height: parent.height
            width: 1
        }
        Rectangle{
            color: "#3f3f3f"
            x: 2
            height: parent.height
            width: 1
        }
    }

    Item {
        id: settingItem
        z: 3
        width: parent.width - notificationItem.width - 40
        height: parent.height - 12
        anchors.top: parent.top
        anchors.topMargin: 10
        anchors.left: verticalSpliter.right
        anchors.leftMargin: 20

        Item {
            id: timeItem
            width: parent.width
            height: parent.height / 5
            anchors.left: parent.left
            anchors.top: parent.top

            Text {
                width: parent.width
                color: "#929292"
                font.pointSize: 10
                text: timeText
            }
        }

        Item {
            id: deleteItem
            width: 40
            height: 40
            anchors.right: parent.right
            anchors.rightMargin: 10
            anchors.bottom: parent.bottom
            anchors.bottomMargin: 10
            visible: false
            z: 2

                Image {
                    id: deleteImage
                    width: parent.width
                    height: parent.height
                    source: "../../resources/icon-close-black.png"
                    fillMode: Image.PreserveAspectFit
                    anchors.centerIn: parent
                }

                Image {
                    id: hoveredImage
                    width: parent.width
                    height: parent.height
                    source: "../../resources/notify_hdelete.png"
                    fillMode: Image.PreserveAspectFit
                    anchors.centerIn: parent
                    visible: false
                }

                MouseArea {
                    z: 5
                    anchors.fill: parent
                    hoverEnabled: true
//                    onHoveredChanged: {
//                        isDeleteHovered =  !isDeleteHovered;
//                        if(isDeleteHovered){
//                            console.log("entered..")
//                            hoveredRectangle.visible = true;
//                            hoveredImage.visible = true;
//                            //deleteImage.source = "../../resources/notify_hdelete.png"
//                        }else{
//                            console.log("exited..");
//                            hoveredRectangle.visible = false;
//                            deleteImage.visible = false;
//                            //deleteImage.source = "../../resources/icon-close-black.png"
//                        }
//                    }

                    onEntered:
                    {
                        deleteItem.visible = true;
                        hoveredRectangle.visible = true;
                        timeItem.visible = false;
                        deleteImage.source = "../../resources/notify_hdelete.png"
                    }

                    onExited: {
                        deleteItem.visible = false;
                        hoveredRectangle.visible = false;
                        timeItem.visible = true;
                        deleteImage.source = "../../resources/icon-close-black.png"
                    }

                    onClicked: {
                        qmlMainWindow.removeNotification(index);
                         notifyList.remove(index);
                    }
                }
        }
    }

    MouseArea {
        id: mouseArea
        anchors.fill: parent
        hoverEnabled: true
        onEntered: {
            hoveredRectangle.visible = true
            deleteItem.visible = true;
            timeItem.visible = false;
        }
        onExited: {
            hoveredRectangle.visible = false;
            deleteItem.visible = false;
            timeItem.visible = true;
        }
//        onHoveredChanged: {
//            isHovered = !isHovered;
//            if(isHovered){
//                hoveredRectangle.visible = true
//                deleteItem.visible = true;
//            }else{
//                hoveredRectangle.visible = false;
//               deleteItem.visible = false;
//            }
//        }
    }

}
