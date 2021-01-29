import QtQuick 2.5
import QtQuick.Dialogs 1.2
Item{
    id: imageListitem
    width: root.width
    height: 30
    property bool           checkSelect: false
    property string         title: "title"
    property string         titleTextColor
    property bool           isHovered: false
    property bool           checkFileName: false
    property int            clickCnt: 0
    property var            selectImage
    property var            imageUrl
    property int             stereoLeftCnt: 0
    property int             stereoRightCnt: 0
    property string          stereoLColor
    property string          stereoRColor
    property bool            isStereoLeft: false
    property bool            isStereoRight: false
    property int             tempStereoType: 0
    property bool            isClicked: true
    property int             spacing: 50
    property string          prefixStr: ""

    Rectangle {
        id: hoveredRectangle
        x: 20
        y: -5
        width: root.width - 50
        height: parent.height + 10
        color: "#1f1f1f"
        visible: false

    }

    Rectangle {
        id: selectRectangle
        x: 20
        y: -5
        width: root.width - 50
        height: parent.height + 10
        color: "#0e3e64"
        visible: checkSelect
    }
    Item {
        id: imageItem
        x: 20
        y: (parent.height - height) / 2
        z: 5
        width: 30
        height: 30

        Image {
            width: 30
            height: 30
            source: "../../resources/icon_image_small.png"
            x:(parent.width - width) / 2
            y: (parent.height - height) / 2
            z: 1
            fillMode: Image.PreserveAspectFit
        }

        MouseArea {
            id: imageArea
            anchors.fill: parent
            z: 10
            hoverEnabled: true
            onEntered: hoveredRectangle.visible = true
            onExited: hoveredRectangle.visible = false

            onClicked: {
                imageIndex = index
                openImageFile();
            }
        }
    }

    Text {
        id: titleText
        x: imageItem.x + imageItem.width + 20
        y: (parent.height - height) / 2
        z: 1
        color: titleTextColor
        text: qsTr(title)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
    }

    Rectangle {
        id: spliterRectangle
        width: 2
        height: parent.height
        color: "#8a8a8a"
        anchors.left: titleText.right
        anchors.leftMargin: 25
        visible: prefixStr === "" ? false: true
    }

    Text {
        id: prefixText
        y: (parent.height - height) / 2
        z: 1
        visible: checkSelect
        anchors.left: spliterRectangle.right
        anchors.leftMargin: 20
        color: titleTextColor
        text: qsTr(prefixStr)
        verticalAlignment: Text.AlignVCenter
        horizontalAlignment: Text.AlignLeft
        font.pixelSize: 15
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
            onEntered: hoveredRectangle.visible = true
            onExited: hoveredRectangle.visible = false
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
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 3;
                    }
                    else
                    {
                        stereoLColor = "#ffffff"
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 1;
                    }
                }
                else
                {
                    if(isStereoRight)
                    {
                        stereoLColor = "#8a8a8a"
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 2;
                    }
                    else
                    {
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 0;
                        stereoLColor = "#8a8a8a"
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
            onEntered: hoveredRectangle.visible = true
            onExited: hoveredRectangle.visible = false
            onClicked: {
				if (root.isRigTemplateMode())
					return;
                stereoRightCnt=stereoRightCnt + 1
                isStereoRight = !isStereoRight
                if (isStereoRight)
                {
                    if(isStereoLeft)
                    {
                        stereoRColor = "#ffffff"
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 3;
                    }
                    else
                    {
                        stereoRColor = "#ffffff"
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 2;
                    }
                }
                else
                {
                    if(isStereoLeft)
                    {
                        stereoRColor = "#8a8a8a"
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 1;
                    }
                    else
                    {
                        selectImage = imageList.get(index);
                        selectImage.stereoType = 0;
                        stereoRColor = "#8a8a8a"
                    }
                }
            }
        }
    }

    Item{
        id: settingItem
        anchors.right: parent.right
        anchors.rightMargin: 60
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

        MouseArea {
            id: settingMouseArea
            z: 0
            anchors.fill: parent
            hoverEnabled: true
            onHoveredChanged: {

            }
            onClicked: {
                globalSettingbox.state = "collapsed";
                var curIndex = index;
                imageTempCamSetting.y = titleRectangle.height +  (index + 1)* 50
                if(imageTempCamSetting.state === "expanded"){
                    imageTempCamSetting.state = "collapsed";

                }else if(imageTempCamSetting.state === "collapsed"){
                    imageTempCamSetting.state = "expanded";
                     getCameraSettings(curIndex);
                     setCameraSettings(curIndex);

                }
            }
        }
    }


    MouseArea {
        id: mouseArea
        z: 2
        width: parent.width * 0.5
        height: parent.height
        hoverEnabled: true
        onHoveredChanged: {
            if(titleText.text === "Empty Slot") return;
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
            imageTempCamSetting.state = "collapsed";
            if(titleText.text === "Empty Slot") return;
             isClicked = !isClicked;
            if(isClicked){
                selectImage = imageList.get(index);
                selectImage.isSelect = true;
                titleText.color = "#ffffff";
                selectRectangle.visible = true;
                settingItem.visible = true;
                stereoItem.visible = true;
                stereoLeftItem.visible = true;
                stereoRightItem.visible = true;
                spliterRectangle.visible = true;
                prefixText.visible = true;

            }else{
                selectImage = imageList.get(index);
                selectImage.isSelect = false;
                spliterRectangle.visible = false;
                prefixText.visible = false;
                checkSelect = false;
                clearImageSettings();
            }
        }
    }

    function setDialogUrl(imageUrl){
        imageDialog.folder = imageUrl;
    }

    function clearImageSettings(){
        selectRectangle.visible = false;
        titleText.color = "#8a8a8a";
        settingItem.visible = false;
        stereoItem.visible = false;
        stereoLeftItem.visible = false;
        stereoRightItem.visible = false;
    }
    function openImageFile(){
        singleImageDialog.open();
    }
}
