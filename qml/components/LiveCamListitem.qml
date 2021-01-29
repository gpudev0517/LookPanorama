import QtQuick 2.5
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4

Item{
	id: videoListitem
    width:  root.width
    height: 30

    property string          title: "title"
    property string          titleTextColor
    property bool            isHovered: false
    property bool            isSelected: false
    property int             clickCnt: 0
    property string          mix: ""
    property bool            checkSelect: false
    property var             selectCamera
    property int             stereoLeftCnt: 0
    property int             stereoRightCnt: 0
    property string          stereoLColor
    property string          stereoRColor
    property int             liveStereoType: 0
    property bool            isClicked: checkSelect
    property bool            isStereoLeft: false
    property bool            isStereoRight: false
    property int             spacing: 50
    property bool            checkFootage: false
    property string          footageImagePath: ""

    Rectangle {
        id: hoveredRectangle
        x: 20
        y: -5
        width: cameraListView.width - 50
        height: parent.height + 10
        color: "#1f1f1f"
        visible: false

    }
    Rectangle {
        id: selectRectangle
        x: 20
        y: -5
        width: cameraListView.width - 50
        height: parent.height + 10
        color: "#0e3e64"
        visible: checkSelect
    }

    Image {
        id: iconImage
        width: 30
        height: 30
        source: "../../resources/icon_camera_small.png"
        x: 50
        y: 0
        z: 1
        fillMode: Image.PreserveAspectFit
    }

    Text {
        id: titleText
        x: iconImage.x + iconImage.width + 20
        y: (parent.height - height) / 2
        z: 1
        color: titleTextColor
        text: qsTr(title)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
    }

    Text {
        id:backgroundfootageText
        x: parent.width / 3
        y: (parent.height - height) / 2
        z: 1
        color: titleTextColor
        text: qsTr("Background Feed")
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
        visible: checkSelect
    }

    RadioButton {
        id: footageRadiobutton
        y: (parent.height - height) / 2
        anchors.left: backgroundfootageText.right
        anchors.leftMargin: 20
        visible: checkSelect
        checked: checkFootage

        onClicked: {
            nodalListView.initfootageVideo()
            nodalListView.initfootageImage()

            selectCamera = camList.get(index)
            selectCamera.isFootage = checked

            var tempFootage = false
            for (var i = 0; i < camList.count; i++) {
                if (camList.get(i).isFootage)
                    tempFootage = true
            }

            if (tempFootage)
                nodalListView.enableFootage = false
            else
                nodalListView.enableFootage = true
        }

       //exclusiveGroup: camListExclusiveGroup

        style: RadioButtonStyle {
               indicator: Rectangle {
                   color: "#171717"
                   implicitWidth: 15
                   implicitHeight: 15
                   radius: 9
                   border.color: "#ffffff"
                   border.width: 1
                   Rectangle {
                       anchors.fill: parent
                       visible: control.checked
                       color: "#ffffff"
                       radius: 9
                       anchors.margins: 4
                   }
               }
        }
    }

    Rectangle {
        id: spliterRectangle
        width: 2
        height: parent.height
        color: "#8a8a8a"
        anchors.left: footageRadiobutton.right
        anchors.leftMargin: 10
        visible: checkSelect && footageRadiobutton.checked ? true: false
    }

    Image {
        id: footageImage
        width: 30
        height: 30
        source: "../../resources/icon_image_small.png"
        anchors.left: spliterRectangle.right
        anchors.leftMargin: 10
        z: 1
        fillMode: Image.PreserveAspectFit
        visible: checkSelect && footageRadiobutton.checked ? true: false

        MouseArea {
            id: imagePathMouseArea
            anchors.fill: parent
            z: 2
            hoverEnabled: true
            onClicked: {
                footageimageDialog.camIndex = index;
                footageimageDialog.open();
            }
        }
    }

    Text {
        id: imagefilePathLabel
        anchors.left: footageImage.right
        anchors.leftMargin: 10
        y: (parent.height - height) / 2
        z: 1
        color: titleTextColor
        text: footageImagePath.length > 30 ? footageImagePath.substring(0,27) + "...": footageImagePath
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
        visible: checkSelect && footageRadiobutton.checked ? true: false
        width: 60
    }

    Image {
        width: 30
        height: 30
        source: "../../resources/uncheck.png"
        anchors.right: stereoItem.left
        anchors.rightMargin: 20
        z: 1
        fillMode: Image.PreserveAspectFit
        visible: checkSelect && footageRadiobutton.checked ? true: false

        MouseArea {
            id: imageclearMouseArea
            anchors.fill: parent
            z: 2
            hoverEnabled: true
            onClicked: {
                cameraListView.updateFootageImagePath(index,"Background weight map");
            }
        }
    }

    Item{
        id: stereoItem
        x: parent.width * 0.7
        y: (parent.height  - height) / 2
        anchors.right: stereoLeftItem.left
        anchors.rightMargin: spacing / 3
        z: 5
        width: 30
        height: 30
        visible: checkSelect

        Image {
            id: stereoImage
            width: 25
            height: 25
            source: "../../resources/ico_camera.png"
            x:(parent.width - width) / 2
            y: (parent.height - height) / 2
            fillMode: Image.PreserveAspectFit
        }
    }

    Item{
        id: stereoLeftItem
        x: parent.width * 0.85
        y: (parent.height - height) / 2
        anchors.right: stereoRightItem.left
        anchors.rightMargin: 50
        width: 40
        height: 30
        z: 1
        visible: checkSelect

        Text{
            id: stereoLeft
            text: "Left"
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            color: stereoLColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            font.pixelSize: 15
        }

        MouseArea {
            id: stereoLeftMouseArea
            z: 1
            anchors.fill: parent
            hoverEnabled: true
            onHoveredChanged: {

                isHovered = !isHovered
                if(isHovered) {
                    hoveredRectangle.visible = true
                    //cursorShape = Qt.PointingHandCursor
                }
                else {
                    hoveredRectangle.visible = false
                }
            }
            onClicked: {
				if (root.isRigTemplateMode())
					return;

               stereoLeftCnt=stereoLeftCnt + 1
                isStereoLeft = !isStereoLeft
                if (isStereoLeft)
                {
                    if(isStereoRight)
                    {
                        stereoLColor = "#ffffff"
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 3;
                        //qmlMainWindow.setTempStereoType(index, stereoType)
                    }
                    else
                    {
                        //stereoLeft.color ="#ffffff";
                        stereoLColor = "#ffffff"
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 1;
                        //qmlMainWindow.setTempStereoType(index, stereoType)
                    }
                }
                else
                {
                    if(isStereoRight)
                    {
                        //stereoLeft.color = "#8a8a8a";
                        stereoLColor = "#8a8a8a"
                        stereoType = 2;
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 2;
                        //qmlMainWindow.setTempStereoType(index, stereoType)
                    }
                    else
                    {
                        stereoType = 0;
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 0;
                        //stereoLeft.color = "#8a8a8a";
                        stereoLColor = "#8a8a8a"
                        //qmlMainWindow.setTempStereoType(index, stereoType)
                    }
                }
            }
        }
    }

    Item{
        id: stereoRightItem
        anchors.right: settingItem.left
        anchors.rightMargin: spacing / 2.5
        width: 40
        height: 30
        z: 1
        visible: checkSelect

        Text{
            id: stereoRight
            text: "Right"
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            color: stereoRColor
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            font.pixelSize: 15

        }

        MouseArea {
            id: stereoRightMouseArea
            z: 1
            anchors.fill: parent
            hoverEnabled: true
            onHoveredChanged: {
                isHovered = !isHovered
                if(isHovered) {
                    hoveredRectangle.visible = true
                    //cursorShape = Qt.PointingHandCursor
                }
                else {
                    hoveredRectangle.visible = false
                }
            }
            onClicked: {
				if (root.isRigTemplateMode())
					return;
                stereoRightCnt=stereoRightCnt + 1
                isStereoRight = !isStereoRight
                if (isStereoRight)
                {
                    if(isStereoLeft)
                    {
                      // stereoRight.color = "#ffffff";
                        stereoRColor = "#ffffff"
                        stereoType = 3;
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 3;
                        //qmlMainWindow.setTempStereoType(index, stereoType)
                    }
                    else
                    {
                        //stereoRight.color ="#ffffff";
                        stereoRColor = "#ffffff"
                        stereoType = 2;
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 2;
                        //qmlMainWindow.setTempStereoType(index, stereoType);
                    }
                }
                else
                {
                    if(isStereoLeft)
                    {
                        //stereoRight.color = "#8a8a8a";
                        stereoRColor = "#8a8a8a"
                        stereoType = 1;
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 1;
                        //qmlMainWindow.setTempStereoType(index, stereoType);
                    }
                    else
                    {
                        stereoType = 0;
                        selectCamera = camList.get(index);
                        selectCamera.stereoType = 0;
                        stereoRColor = "#8a8a8a"
                        //qmlMainWindow.setTempStereoType(index, stereoType);
                    }
                }
            }
        }
    }

    Item{
        id: settingItem
        anchors.right: arrowUpItem.left
        anchors.rightMargin: 15
        z: 5
        width: 40
        height: 30
        visible: checkSelect

        Rectangle {
            id: settingHoveredRectangle
            anchors.fill: parent
            color:"#c75050"
            z: 0
            visible: false
        }

        Image {
            id: settingImage
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            source: "../../resources/setting.png"
            z: 1
            fillMode: Image.PreserveAspectFit


        }
    }

    Item{
        id: arrowUpItem
        anchors.right: arrowDownItem.left
        anchors.rightMargin: 1
        z: 5
        width: 20
        height: 30
        visible: checkSelect & index > 0 ? true : false

        Rectangle {
            id: upHoveredRectangle
            anchors.fill: parent
            color:"#353535"
            visible: false
        }

        Image {
            id: upImage
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 20
            height: 25
            source: "../../resources/ico_moveup.png"
            z: 1
            fillMode: Image.PreserveAspectFit

            MouseArea {
                id: upArea
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    camList.move(index, index - 1, 1);
                }
            }
        }
    }

    Item{
        id: arrowDownItem
        anchors.right: parent.right
        anchors.rightMargin: 50
        z: 5
        width: 20
        height: 30
        visible: (checkSelect & index < camList.count-1) ? true : false

        Rectangle {
            id: downHoveredRectangle
            anchors.fill: parent
            color:"#353535"
            visible: false
        }

        Image {
            x: (parent.width - width) / 2
            y: (parent.height - height) / 2
            width: 20
            height: 25
            source: "../../resources/ico_movedown.png"
            z: 1
            fillMode: Image.PreserveAspectFit

            MouseArea {
                anchors.fill: parent
                hoverEnabled: true
                onClicked: {
                    camList.move(index, index + 1, 1);
					getInfoStr();
                }
            }
        }
    }

    MouseArea {
        id: mouseArea
        x: 0
        z: 2
        width: parent.width * 0.3
        height: parent.height
        hoverEnabled: true
        onHoveredChanged: {
            cursorShape = Qt.PointingHandCursor
            isHovered = !isHovered
            if(isHovered) {
                hoveredRectangle.visible = true
            }
            else {
                hoveredRectangle.visible = false
            }
        }
        onClicked:{
            globalSettingbox.state = "collapsed";
            liveTempCamSetting.state = "collapsed";
            isClicked = !isClicked;
            if(isClicked){
                titleText.color = "#ffffff"
                selectCamera = camList.get(index);
                settingItem.visible = true
                selectCamera.isSelect = true;
                selectRectangle.visible = true;
            }else{
                selectCamera = camList.get(index);
                selectCamera.isSelect = false;
                selectCamera.stereoType = 0;
                isStereoLeft = false;
                isStereoRight = false;
                stereoLColor = "#8a8a8a"
                stereoRColor = "#8a8a8a";
                titleText.color = "#8a8a8a";
                selectRectangle.visible = false;
                settingItem.visible = false;
            }
        }
    }
}
