import QtQuick 2.0

Item {
    width: 255
    height: 48
    property bool isHoveredCheck: false
    property bool isHoveredUncheck: false
    property bool isHoveredMore: false
    property bool isHover: false

    Rectangle {
        id: backgroundRectangle
        width: parent.width
        height: parent.height
        color: "#1f1f1f"
    }

    Item{
        id: oculusItem
        anchors.right: spliter.left
        anchors.rightMargin: 15
        width: 68
        height: 48
        visible: true
        Rectangle{
            id: exposureHoverRectangle
            width: parent.width
            height: parent.height
            color: "#353535"
            visible: false
        }

        Image {
            z: 1
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            fillMode: Image.PreserveAspectFit
            source: "../../resources/icon_oculus.png"
        }

    }

    Item {
        id: spliter
        y: (parent.height - height) / 2
        width: 2
        height: parent.height - 20
        anchors.right: checkItem.left
        anchors.rightMargin: 15
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
        id: checkItem
        anchors.right: uncheckItem.left
        width: 68
        height: 48
        z: 2
        Rectangle {
            id: checkHoveredRectangle
            width: parent.width
            height: 48
            color: "#353535"
            visible: false
        }

        Image {
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 2
            fillMode: Image.PreserveAspectFit
            source: "../../resources/check.png"
        }
        MouseArea {
            z: 2
            anchors.fill: parent
            hoverEnabled: isHover
            onHoveredChanged: {
                isHoveredCheck = !isHoveredCheck
                if(isHoveredCheck)
                    checkHoveredRectangle.visible = true
                else
                    checkHoveredRectangle.visible = false

            }
            onClicked:
            {
                toolbox.clearSelected();
                statusItem.setPlayAndPause();
                qmlMainWindow.enableOculus(true);
                oculusGroup.isHover = false;
                oculusGroup.state = "collapsed";
                oculusTopCtrl.visible = true;
            }
        }

    }

    Item {
        id:　uncheckItem
        anchors.right: parent.right
        width:　68
        height: 48
        z: 2
        Rectangle {
            id: unCheckHoveredRectangle
            width: parent.width
            height: 48
            color: "#353535"
            visible: false
        }
        Image {
            x: (parent.width - width) / 2
            y: (parent.heigth - height) / 2
            anchors.verticalCenter: parent.verticalCenter
            fillMode: Image.PreserveAspectFit
            source: "../../resources/uncheck.png"
        }
        MouseArea {
            z: 2
            anchors.fill: parent
            hoverEnabled: isHover
            onHoveredChanged: {
                isHoveredUncheck = !isHoveredUncheck
                if(isHoveredUncheck)
                {
                    unCheckHoveredRectangle.visible = true;
                }
                else
                    unCheckHoveredRectangle.visible = false
            }
//            onEntered: unCheckHoveredRectangle.visible = true;
//            onExited: unCheckHoveredRectangle.visible = false;
             onClicked:{
                 toolbox.clearSelected();
                 statusItem.setPlayAndPause();
                 qmlMainWindow.enableOculus(false);
                 oculusGroup.isHover = false;
                 oculusGroup.state = "collapsed";
                 oculusTopCtrl.visible = true;
             }
        }
    }

}

