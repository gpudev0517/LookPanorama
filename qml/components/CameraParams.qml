import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import "../controls"

Item {
    width : 330
    height: 500

    property bool       isHovered : false
    property int        leftMargin: 20
    property int        rightMargin: 0
    property int        spacing: 20
    property color      textColor: "#ffffff"
    property color      spliterColor: "#555555"
    property int        lblFont: 14
    property int        itemHeight:30
    property bool       isCPoint: false
    property int        cameraCnt
    property var        opacityVal: 0.9
    property int        camWidth: 100
    property int        camHeight: 100

    ScrollView {
        id: scrollView
        width: parent.width
        height: parent.height
        verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        horizontalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        flickableItem.interactive: true

        style: ScrollViewStyle {
            transientScrollBars: true
            handle: Item {
                implicitWidth: 14
                implicitHeight: 26
                Rectangle {
                    color: "#424246"
                    anchors.fill: parent
                    anchors.topMargin: 6
                    anchors.leftMargin: 4
                    anchors.rightMargin: 4
                    anchors.bottomMargin: 6
                }

            }

            scrollBarBackground: Item {
                implicitWidth: 14
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

        Item{
            id: groupItems
            width: scrollView.width
            height: parent.height

			function convertAngle(angleText)
			{
				var val = Number(angleText);
				while (val > 180)
					val = val - 360;
				while(val < -180)
					val = val + 360;
                return val;
			}

			Item {
                id: lensTypeItem
                width: parent.width
                height: 30
                anchors.top: parent.top
                anchors.topMargin: 5

                Text {
                    id: lensTypeText
                    color: textColor
                    width: lblWidth
                    text: qsTr("Lens Type")
                    horizontalAlignment: Text.AlignLeft
                    anchors.left: parent.left
                    anchors.leftMargin: spacing
                    anchors.verticalCenter: parent.verticalCenter
                    font.pixelSize: 13
                }

                ComboBox {
                    id: lensTypeCombo
                    width: parent.width / 2
                    height: 30
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: lensTypeText.right
                    anchors.leftMargin: spacing
                    model: ["Standard", "Fullframe Fisheye", "Circular Fisheye", "CV Standard", "CV Fisheye"]
					enabled: false
                    onCurrentTextChanged: {					
						qmlMainWindow.setLensType(currentIndex);
                        getCameraParams();
                    }
                }
            }

            Text {
                id: yawLabel
                text: qsTr("Yaw")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　yawItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: yawItem
                width: parent.width
                height: 30
                anchors.top: lensTypeItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: yawSlider
                    value: yawSpin.text
                    width: parent.width * 0.7 - leftMargin * 2
                    minimumValue: -180
                    maximumValue: 180
                    stepSize: 0.01
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }
                }

                FlatSpin {
                    id: yawSpin
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    maximumValue: yawSlider.maximumValue
                    minimumValue: yawSlider.minimumValue
                    value: yawSlider.value                    
                    onEditingFinished: {
                        yawSlider.value = groupItems.convertAngle(yawSpin.value);
                        setCameraParams();
                    }
                    onValueChanged: {
                        yawSlider.value = groupItems.convertAngle(yawSpin.value);
                        setCameraParams();
                    }
				}				
            }

            Text {
                id: pitchLabel
                text: qsTr("Pitch")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　pitchItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: pitchItem
                width: 330
                height: 30
                anchors.top: yawItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: pitchSlider
                    value: pitchText.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -180
                    maximumValue: 180
                    stepSize: 0.01
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }

                }

                FlatText {
                    id: pitchText
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: pitchSlider.value
                    maximumLength: 6
                    onEditingFinished: {
                        pitchSlider.value = groupItems.convertAngle(pitchText.text);
                        setCameraParams();
                    }
                }
            }

            Text {
                id: rollLabel
                text: qsTr("Roll")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　rollItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: rollItem
                width: 330
                height: 30
                anchors.top: pitchItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: rollSlider
                    value: rollText.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -180
                    maximumValue: 180
                    stepSize: 0.01
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }

                }

                FlatText {
                    id: rollText
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: rollSlider.value
                    maximumLength: 6
                    onEditingFinished: {
						rollSlider.value = groupItems.convertAngle(rollText.text);
                        setCameraParams();
                    }
                }
            }

            Text {
                id: fovLabel
                text: qsTr("Fov")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　fovItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: fovItem
                width: 330
                height: 30
                anchors.top: rollItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: fovSlider
                    value: fovText.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: 0.1
                    maximumValue: 360
                    stepSize: 0.01
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }

                }

                FlatText {
                    id: fovText
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: fovSlider.value
                    maximumLength: 6
//                    onAccepted: {
//                        fovSlider.value = fovText.text;
//                        setCameraParams();
//                    }

                    onEditingFinished: {
                        fovSlider.value = fovText.text;
                        setCameraParams();
                    }
                }
            }

			Text {
                id: fovYLabel
                text: qsTr("FocalY")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　fovYItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: fovYItem
                width: 330
                height: 30
                anchors.top: fovItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: fovYSlider
                    value: fovYText.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: 0.1
                    maximumValue: 360
                    stepSize: 0.01
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }
                }

                FlatText {
                    id: fovYText
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: fovYSlider.value
                    maximumLength: 6
//                    onAccepted: {
//                        fovYSlider.value = fovYText.text;
//                        setCameraParams();
//                    }

                    onEditingFinished: {
                        fovYSlider.value = fovYText.text;
                        setCameraParams();
                    }
                }
            }

            Text {
                id: k1Label
                text: qsTr("K1")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　k1Item.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: k1Item
                width: 330
                height: 30
                anchors.top: fovYItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: k1Slider
                    value: k1Text.text * 100
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -2
                    maximumValue: 2
                    stepSize: 0.01
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }

                }

                FlatText {
                    id: k1Text
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: k1Slider.value
                    maximumLength: 6
//                    onAccepted: {
//                        k1Slider.value = k1Text.text;
//                        setCameraParams();
//                    }

                    onEditingFinished: {
                        k1Slider.value = k1Text.text;
                        setCameraParams();
                    }

                }
            }

            Text {
                id: k2Label
                text: qsTr("K2")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　k2Item.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: k2Item
                width: 330
                height: 30
                anchors.top: k1Item.bottom
                anchors.topMargin: spacing

                Slider {
                    id: k2Slider
                    value: k2Text.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -2
                    maximumValue: 2
                    stepSize: 0.001
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }
                }

                FlatText {
                    id: k2Text
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: k2Slider.value
                    maximumLength: 6
//                    onAccepted: {
//                        k2Slider.value = k2Text.text;
//                        setCameraParams();
//                    }

                    onEditingFinished: {
                        k2Slider.value = k2Text.text;
                        setCameraParams();
                    }
                }
            }

            Text {
                id: k3Label
                text: qsTr("K3")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　k3Item.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: k3Item
                width: 330
                height: 30
                anchors.top: k2Item.bottom
                anchors.topMargin: spacing

                Slider {
                    id: k3Slider
                    value: k3Text.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -2
                    maximumValue: 2
                    stepSize: 0.0001
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }

                }

                FlatText {
                    id: k3Text
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: k3Slider.value
                    maximumLength: 7
//                    onAccepted: {
//                        k3Slider.value = k3Text.text;
//                        setCameraParams();
//                    }

                    onEditingFinished: {
                        k3Slider.value = k3Text.text;
                        setCameraParams();
                    }
                }
            }

            Text {
                id: offsetXLabel
                text: qsTr("OffsetX")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　offsetXItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: offsetXItem
                width: 330
                height: 30
                anchors.top: k3Item.bottom
                anchors.topMargin: spacing

                Slider {
                    id: offsetXSlider
                    value: offsetXText.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -camWidth / 2
                    maximumValue: camWidth / 2
                    stepSize: 0.1
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }

                }

                FlatText {
                    id: offsetXText
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: offsetXSlider.value
                    maximumLength: 6
//                    onTextChanged: {
//                        offsetXSlider.value = offsetXText.text;
//                        setCameraParams();
//                    }

                    onEditingFinished: {
                        offsetXSlider.value = offsetXText.text;
                        setCameraParams();
                    }
                }
            }

            Text {
                id: offsetYLabel
                text: qsTr("OffsetY")
                verticalAlignment: Text.AlignVCenter
                color: "#ffffff"
                font.pixelSize: lblFont
                anchors.bottom:　offsetYItem.top
                anchors.left: parent.left
                anchors.leftMargin: leftMargin
            }

            Item {
                id: offsetYItem
                width: 330
                height: 30
                anchors.top: offsetXItem.bottom
                anchors.topMargin: spacing

                Slider {
                    id: offsetYSlider
                    value: offsetYText.text
                    width: parent.width * 0.85 - leftMargin * 2
                    minimumValue: -camHeight / 2
                    maximumValue: camHeight/ 2
                    stepSize: 0.1
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    updateValueWhileDragging: true
                    onValueChanged: {
                        if (pressed) {
                            setCameraParams();
                        }
                    }
                }

                FlatText {
                    id: offsetYText
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    width: parent.width * 0.15
                    text: offsetYSlider.value
                    maximumLength: 6

                    onEditingFinished: {
                        offsetYSlider.value = offsetYText.text;
                        setCameraParams();
                    }
                }
            }

//            Item {
//                id: applyItem
//                width: 70
//                height: 30
//                anchors.top: offsetYItem.bottom
//                anchors.topMargin: spacing
//                anchors.left: parent.left
//                anchors.leftMargin: leftMargin

//                Rectangle {
//                    id: applyHoverRect
//                    width: parent.width
//                    height: parent.height
//                    color: "#373737"
//                    visible: false
//                    border.color: "#4e8d15"
//                    z: 1

//                    Text {
//                        color: "#ffffff"
//                        text: "Apply"
//                        font.pointSize: 11
//                        horizontalAlignment: Text.AlignHCenter
//                        verticalAlignment: Text.AlignVCenter
//                        anchors.fill: parent
//                    }
//                }

//                Rectangle {
//                    id: applyRect
//                    width: parent.width
//                    height: parent.height
//                    color: "#373737"

//                    Text {
//                        id: applyText
//                        color: "#ffffff"
//                        text: "Apply"
//                        font.pointSize: 11
//                        horizontalAlignment: Text.AlignHCenter
//                        verticalAlignment: Text.AlignVCenter
//                        anchors.fill: parent


//                    }

//                    MouseArea {
//                        id: applyMouseArea
//                        anchors.fill: parent
//                        hoverEnabled: true
//                        onHoveredChanged: {
//                            isHovered = !isHovered
//                            if(isHovered){
//                                applyHoverRect.visible = true;
//                            }else{
//                                applyHoverRect.visible = false;
//                            }
//                        }

//                        onClicked: {
//                            setCameraParams();
//                        }
//                    }
//                }
//            }

        }
    }

    function getCameraParams(){
        camWidth = qmlMainWindow.getWidth(cameraCombo.currentIndex);
        camHeight = qmlMainWindow.getHeight(cameraCombo.currentIndex)
        yawSlider.value = qmlMainWindow.getYaw(cameraCombo.currentIndex);
        pitchSlider.value = qmlMainWindow.getPitch(cameraCombo.currentIndex);
        rollSlider.value = qmlMainWindow.getRoll(cameraCombo.currentIndex);
        fovSlider.value = qmlMainWindow.getFov(cameraCombo.currentIndex);
		fovYSlider.value = qmlMainWindow.getFovy(cameraCombo.currentIndex);
        k1Slider.value = qmlMainWindow.getK1(cameraCombo.currentIndex);
        k2Slider.value = qmlMainWindow.getK2(cameraCombo.currentIndex);
        k3Slider.value = qmlMainWindow.getK3(cameraCombo.currentIndex);
        offsetXSlider.value = qmlMainWindow.getOffsetX(cameraCombo.currentIndex);
        offsetYSlider.value = qmlMainWindow.getOffsetY(cameraCombo.currentIndex);

		// check lensType
		var lensType = qmlMainWindow.getLensType();
		lensTypeCombo.currentIndex = lensType;
		if (lensType == 3) {
			// LensType_opencvLens_Standard
			fovLabel.text = "FoV";			
			fovYLabel.visible = false;
			fovYItem.visible = false;						 
			k1Item.anchors.top = fovItem.bottom;
		} else if (lensType == 4) {
			// LensType_opencvLens_Fisheye
			fovLabel.text = "FoVX";
			fovLabel.text = "FoVY";
			fovYLabel.visible = true;
			fovYItem.visible = true;
			k1Item.anchors.top = fovYItem.bottom;
		} else {
			fovLabel.text = "FoV";			
			fovYLabel.visible = false;
			fovYItem.visible = false;			 
			k1Item.anchors.top = fovItem.bottom;
		}
    }

    function setCameraParams(){

        if(!qmlMainWindow.start) return;

        qmlMainWindow.setYaw(yawSlider.value,cameraCombo.currentIndex);
        qmlMainWindow.setPitch(pitchSlider.value,cameraCombo.currentIndex);
        qmlMainWindow.setRoll(rollSlider.value,cameraCombo.currentIndex);
        qmlMainWindow.setFov(fovSlider.value,cameraCombo.currentIndex);
		qmlMainWindow.setFovy(fovYSlider.value,cameraCombo.currentIndex);
        qmlMainWindow.setK1(k1Slider.value,cameraCombo.currentIndex);
        qmlMainWindow.setK2(k2Slider.value,cameraCombo.currentIndex);
        qmlMainWindow.setK3(k3Slider.value,cameraCombo.currentIndex);
        qmlMainWindow.setOffsetX(offsetXSlider.value,cameraCombo.currentIndex);
        qmlMainWindow.setOffsetY(offsetYSlider.value,cameraCombo.currentIndex);
        qmlMainWindow.reStitch(true);
    }

    function insertCamName(){
        cameraModel.clear();
        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
            cameraModel.append({"text": "Camera" + (i + 1)})
        }

     }
}
