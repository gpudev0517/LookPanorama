import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import "../controls"

Item {
    id: videoGlobalSettingsItem
    width : 350
    height: 800
    opacity: 1

    property int        leftMargin: 15
    property int        rightMargin: 15
    property color      textColor: "#ffffff"
    property int        lblFont: 14
    property int        itemHeight: 15
    property int        lblWidth: 80
    property int        textWidth: 45
    property int        spacing: 25
    property bool       isPanoramaType
    property bool       isYoutube: false
    property bool       tempYouTube: false
    property bool       isNvidia: false

    property int        youTubeWidth: 3840
    property int        youTubeHeight: 2160
    property int        originPanoramaWidth
    property int        originPanoramaHeight

	property var templateModes: {
	  "LIVE": {"value": 0, "name": "Live Camera Setting"},
	  "VIDEO": {"value": 1, "name": "Video Setting"},
	  "IMAGE": {"value": 2, "name": "Frame Sequence Setting"},
	  "NONE": {"value": -1, "name": ""}
	};

	property var		settingMode : templateModes.NONE;

    Rectangle {
        id: backgroundRectangle
        x: 0
        y: 0
        width: parent.width
        height: parent.height
        color: "#171717"
        opacity: 0.9
    }

    Rectangle {
        id: titleRectangle
        x: 0
        y: 0
        width: parent.width
        height: 48
        color: "#171717"
        z: 1
        opacity: 1

        Text {
            id: titleText
			text: settingMode.name
            x: (350 - width) / 2
            y: (parent.height - height) / 2
            z: 3
            color: "#ffffff"
            font.bold: false
            font.pixelSize: 20
        }
    }
    

    Spliter {
        id: spliterRectangle
        width: parent.width
        height: 2
        z: 3
        anchors.top: titleRectangle.bottom
    }

    ScrollView {
        id: scrollView
        y: titleRectangle.height
        width: parent.width
        height: parent.height
        horizontalScrollBarPolicy: 1
        opacity: 0.8
        verticalScrollBarPolicy: Qt.ScrollBarAlwaysOff
        flickableItem.interactive: true

        style: ScrollViewStyle {
            transientScrollBars: false
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
                //implicitHeight: leftMargin
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
            //handleOverlap: 70

        }
        Item{
            id: groupItems
            width: scrollView.width
            height: itemHeight * 80

            Item {
                id: resolutionItem
                width: parent.width
                height: 30
                anchors.top: parent.top
                anchors.topMargin: itemHeight
                Text {
                    id:xText
                    color: textColor
                    width: lblWidth
                    text: qsTr("Width")
                    horizontalAlignment: Text.AlignLeft
                    anchors.left: parent.left
                    anchors.leftMargin: leftMargin
                    anchors.verticalCenter: parent.verticalCenter
                    font.pixelSize: lblFont

                }

                FlatText {
                    id: xRes
                    width: textWidth
                    height: 30
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.left: xText.right
                    anchors.leftMargin: spacing
                    enabled: true
                }

                Text {
                    id: yText
                    x: 175
                    color: textColor
                    width: lblWidth
                    text: qsTr("Height")
                    horizontalAlignment: Text.AlignLeft
                    anchors.verticalCenter: parent.verticalCenter
                    font.pixelSize: lblFont
                }

                FlatText {
                    id: yRes
                    width: textWidth
                    height: 30
                    anchors.verticalCenter: parent.verticalCenter
                    anchors.right: parent.right
                    anchors.rightMargin: rightMargin
                    enabled: true
                }



                Item {
                    id: panoResItem
                    width: resolutionItem.width
                    height: 30
                    anchors.top: resolutionItem.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id:xPanoText
                        width: lblWidth
                        color: textColor
                        text: qsTr("Panorama Width")
                        textFormat: Text.RichText
                        wrapMode: Text.NoWrap
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: xPano
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: xPanoText.right
                        anchors.leftMargin: spacing
                        onEditingFinished:  {
                            yPano.text = text / 2;
                        }
                    }

                    Text {
                        id: yPanoText
                        x: 175
                        color: textColor
                        width: lblWidth
                        text: qsTr("Panorama Height")
                        textFormat: Text.AutoText
                        wrapMode: Text.NoWrap
                        horizontalAlignment: Text.AlignLeft
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: yPano
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin

                        onEditingFinished:  {
                            xPano.text = text * 2;
                        }
                    }
                 }

                Item {
                    id: singleFrameItem
                    width: resolutionItem.width
                    height: 30
                    anchors.top: panoResItem.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id: singleFrameText
                        width: lblWidth
                        color: textColor
                        text: qsTr("Single Frame")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    RadioButton {
                        id: singleFrameRadio
                        y: (parent.height - height) / 2
                        width: textWidth
                        anchors.left: singleFrameText.right
                        anchors.leftMargin: spacing
                        checked: false

                        onCheckedChanged: {
                            if(singleFrameRadio.checked === true){
                                startFrame.enabled = false;
                                endFrame.enabled = false;
                                startFrame.text = -1;
                                endFrame.text = -1;
                            } else {
                                startFrame.enabled = true;
                                endFrame.enabled = true;
                            }
                        }

                        style: RadioButtonStyle {
                               indicator: Rectangle {
                                   color: "#171717"
                                   implicitWidth: 15
                                   implicitHeight: 15
                                   radius: 9
                                   border.color: "#4e8d15"
                                   border.width: 1
                                   Rectangle {
                                       anchors.fill: parent
                                       visible: control.checked
                                       color: "#4e8d15"
                                       radius: 9
                                       anchors.margins: 4
                                   }
                               }

                           }
                    }
                 }


                Item {
                    id: frameItem
                    width: resolutionItem.width
                    height: 30
                    anchors.top: singleFrameItem.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id: startText
                        width: lblWidth
                        color: textColor
                        text: qsTr("Start")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: startFrame
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: startText.right
                        anchors.leftMargin: spacing

                        onTextChanged: {
                            if(singleFrameRadio.checked === true)
								startFrame.enabled = false;
							else if(singleFrameRadio.checked === false)
								startFrame.enabled = true;

                        }
                    }

                    Text {
                        id: endText
                        x: 175
                        color: textColor
                        width: lblWidth
                        text: qsTr("End ")
                        horizontalAlignment: Text.AlignLeft
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: endFrame
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin

						 onTextChanged: {
                            if(singleFrameRadio.checked === true)
								endFrame.enabled = false;
							else if(singleFrameRadio.checked === false)
								endFrame.enabled = true;
                        }
                    }
                 }

                Item {
                    id: cameraCalibItem
                    x: 0
                    width: parent.width
                    height: 30
                    anchors.top: frameItem.bottom
                    anchors.topMargin: itemHeight

                    Text {
                        id: cameraCalibText
                        color: textColor
                        width: lblWidth
                        text: qsTr("Calibration File")
                        horizontalAlignment: Text.AlignLeft
                        font.pixelSize: lblFont
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                    }

                    FlatText {
                        id: cameraCalib
                        anchors.left: cameraCalibText.right
                        anchors.leftMargin: spacing
                        width: parent.width / 2 - 4
                        height: 30
                        //textFormat: Text.AutoText
                        anchors.verticalCenter: parent.verticalCenter
                        //color: "#4e8d15"
                        //wrapMode: Text.WordWrap

			onEditingFinished: {
				if (text === "") {
					hfov.enabled = true;
					lensTypeCombo.enabled = true;
				} else {
					hfov.enabled = false;
					lensTypeCombo.enabled = false;
				}
			}
                    }
                    Rectangle{
                        width: 30
                        height: 30
                        z: 1
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
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
                        id: fileHoveredRectangle
                        width: 30
                        height: 30
                        z: 1
                        color: "#373737"
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
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
                        x: fileHoveredRectangle.x
                        z: 2
                        width: fileHoveredRectangle.width
                        height: fileHoveredRectangle.height
                        anchors.verticalCenter: parent.verticalCenter
                        hoverEnabled: true
                        onEntered: fileHoveredRectangle.visible = true
                        onExited: fileHoveredRectangle.visible = false
                        onClicked: fileDialog.visible = true
                    }
                }
                FileDialog{
                    id:fileDialog
                    title: "Select calibration file"
                    nameFilters: [ "Calib File (*.pac *.pts)", "All files (*)" ]
                    selectMultiple: false

                    onSelectionAccepted: {
                        var fileName = fileUrl.toString().substring(8); // Remove "file:///" prefix
                        cameraCalib.text = fileName;
                        hfov.enabled = false;
                        lensTypeCombo.enabled = false;
                    }
                }

		Item {
			id: autoCalib
			width: parent.width
			height: 30
			anchors.top: cameraCalibItem.bottom
			anchors.topMargin: itemHeight
			Text {
				id: lensTypeLabel
				width: lblWidth
				color: textColor
				text: qsTr("Lens Type")
				horizontalAlignment: Text.AlignLeft
				anchors.left: parent.left
				anchors.leftMargin: leftMargin
				anchors.verticalCenter: parent.verticalCenter
				font.pixelSize: lblFont

			}

			ComboBox {
				id: lensTypeCombo
				width: parent.width / 4
				height: 30
				anchors.verticalCenter: parent.verticalCenter
				anchors.left: lensTypeLabel.right
				anchors.leftMargin: spacing
				model: ["Standard", "Fullframe Fisheye", "Circular Fisheye", "CV Standard", "CV Fisheye"]
                onCurrentIndexChanged:
                {
                    if (currentIndex)
                        hfov.text = 240;
                    else
                        hfov.text = 120;
                }
			}

			Text {
				id: hfovLabel
				color: textColor
				text: qsTr("HFOV")
				horizontalAlignment: Text.AlignLeft
                anchors.right: hfov.left
                anchors.rightMargin: rightMargin
				anchors.verticalCenter: parent.verticalCenter
				font.pixelSize: lblFont

			}

			FlatText {
			        id: hfov
			        width: textWidth
			        height: 30
			        anchors.verticalCenter: parent.verticalCenter
			        anchors.right: parent.right
			        anchors.rightMargin: rightMargin
			}
		}

                Item {
                    id: fpsItem
                    width: resolutionItem.width
                    height: 30
                    anchors.top: autoCalib.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id: fpsText
                        width: lblWidth
                        color: textColor
                        text: qsTr("FPS")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: fps
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: fpsText.right
                        anchors.leftMargin: spacing
                    }

                    Text {
                        id: sourceFpsText
                        color: textColor
                        text: qsTr("Source FPS")
                        horizontalAlignment: Text.AlignLeft
                        anchors.right: sourceFps.left
                        anchors.rightMargin: rightMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: sourceFps
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                    }

                 }

				 Item {
                    id: lidarPortItem
                    width: resolutionItem.width
                    height: 30
                    anchors.top: fpsItem.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id: lidarPortText
                        width: lblWidth
                        color: textColor
                        text: qsTr("LIDAR PORT")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    FlatText {
                        id: lidarPort
                        width: textWidth
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: lidarPortText.right
                        anchors.leftMargin: spacing
                    }

                 }

                Item {
                    id: fileExtItem
                    width: resolutionItem.width
                    height: 30
                    anchors.top: lidarPortItem.bottom
                    anchors.topMargin: itemHeight
                    Text {
                        id: fileExtText
                        width: lblWidth
                        color: textColor
                        text: qsTr("File Extension")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: leftMargin
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: lblFont

                    }

                    ComboBox {
                        id: fileExtCombo
                        width: parent.width / 3
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: fileExtText.right
                        anchors.leftMargin: spacing
                        model: ["dpx","tiff","jpg","png","bmp"]
                    }

                 }

                Item {
                    id: outItem
                    width: resolutionItem.width
                    height: 70
                    visible: true
                    anchors.top: fileExtItem.bottom
                    anchors.topMargin: itemHeight

                    Text {
                        id: outLabel
                        x: leftMargin
                        y: 5
                        width: lblWidth
                        color: textColor
                        text: qsTr("Output")
                        font.bold: true
                        font.pixelSize: 16
                        horizontalAlignment: Text.AlignLeft

                    }

                    Spliter {
                        id: outSpliter
                        width: parent.width
                        height: 2
                        anchors.top: parent.top
                        anchors.topMargin: itemHeight + 10
                    }

                    Item {
                        id: splitMinsItem
                        width: resolutionItem.width
                        height: 30
                        anchors.top: outSpliter.bottom
                        anchors.topMargin: itemHeight
                        Text {
                            id: splitMinsLabel
                            width: lblWidth
                            color: textColor
                            text: qsTr("Split")
                            horizontalAlignment: Text.AlignLeft
                            anchors.left: parent.left
                            anchors.leftMargin: leftMargin
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont

                        }

                        FlatText {
                            id: splitMin
                            width: textWidth
                            height: 30
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: splitMinsLabel.right
                            anchors.leftMargin: spacing
                        }

                        Text {
                            x: 175
                            color: textColor
                            width: lblWidth
                            text: qsTr("Mins")
                            horizontalAlignment: Text.AlignLeft
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont

                        }

                     }

                    Item {
                        id: qualityItem
                        width: resolutionItem.width
                        height: 30
                        anchors.top: splitMinsItem.bottom
                        anchors.topMargin: itemHeight
                        Text {
                            id: qualityLabel
                            width: lblWidth
                            color: textColor
                            text: qsTr("Quality")
                            horizontalAlignment: Text.AlignLeft
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont
                            anchors.left: parent.left
                            anchors.leftMargin: leftMargin

                        }

                        Text {
                            id: lowText
                            color: textColor
                            horizontalAlignment: Text.AlignLeft
                            anchors.left: qualityLabel.right
                            anchors.leftMargin: spacing
                            text: qsTr("Low")
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont
                        }

                        Slider {
                            id: qualitySlider
                            width: parent.width / 2.6
                            minimumValue: 0
                            maximumValue: 51
                            stepSize: 1
                            anchors.left: lowText.right
                            anchors.leftMargin: 10
                            anchors.verticalCenter: parent.verticalCenter
                        }

                        Text {
                            id: highText
                            text: qsTr("High")
                            width: 30
                            horizontalAlignment: Text.AlignLeft
                            color: textColor
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont
                            anchors.right: parent.right
                            anchors.rightMargin: rightMargin
                        }

                     }

                    Item {
                        id: savePathItem
                        width: 350
                        height: 30
                        anchors.top: qualityItem.bottom
                        anchors.topMargin: 20
                        Text {
                            id: savePathText
                            width: 100
                            y: 12
                            color: textColor
                            text: qsTr("Snapshot Path")
                            anchors.left: parent.left
                            anchors.leftMargin: leftMargin
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont

                        }

                        FlatText {
                            id: savePathFeild
                            width: parent.width * 0.5
                            height: 30
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: savePathText.right
                            anchors.rightMargin: rightMargin
                        }

                        Rectangle{
                            width: 30
                            height: 30
                            z: 1
                            anchors.right: parent.right
                            anchors.rightMargin: rightMargin
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
                            anchors.rightMargin: rightMargin
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
                    }

                    FileDialog{
                        id: saveFileDialoge
                        title: "Select snapshot path"
                        selectMultiple: false
                        selectFolder: true;

                        onSelectionAccepted: {
                            var fileName = fileUrl.toString().substring(8); // Remove "file:///" prefix
                            savePathFeild.text = fileName;
                        }
                    }

                    Item {
                        id: codecItem
                        width: resolutionItem.width
                        height: 30
                        anchors.top: savePathItem.bottom
                        anchors.topMargin: itemHeight
                        Text {
                            id: rtmpvideoCodecLabel
                            width: lblWidth
                            color: textColor
                            text: qsTr("V-Codec")
                            horizontalAlignment: Text.AlignLeft
                            anchors.left: parent.left
                            anchors.leftMargin: leftMargin
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont

                        }

                        ComboBox {
                            id: rtmpVideoCombo
                            width: parent.width / 5
                            height: 30
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.left: rtmpvideoCodecLabel.right
                            anchors.leftMargin: spacing
                            model: ["H.264","H.265"]
                        }

                        Text {
                            id: rtmpaudioCodecLabel
                            color: textColor
                            text: qsTr("A-Codec")
                            horizontalAlignment: Text.AlignLeft
                            anchors.left: rtmpVideoCombo.right
                            anchors.leftMargin: spacing / 2
                            anchors.verticalCenter: parent.verticalCenter
                            font.pixelSize: lblFont
                        }

                        ComboBox {
                            id: rtmpAudioCombo
                            width: parent.width / 5
                            height: 30
                            anchors.verticalCenter: parent.verticalCenter
                            anchors.right: parent.right
                            anchors.rightMargin: rightMargin
                            model: ["AAC"]
                        }
                     }

                    Item {
                        id: nvidiaItem
                        width: parent.width
                        height: 30
                        anchors.top: codecItem.bottom
                        anchors.topMargin: 10
                        Text {
                            id:  nvidiaLabel
                            x: leftMargin
                            width: lblWidth
                            anchors.verticalCenter: parent.verticalCenter
                            color: "#ffffff"
                            text: qsTr("NVidia Encoding")
                            font.pixelSize: lblFont
                        }

                        Switch{
                            id: nvidiaSwitch
                            anchors.left: nvidiaLabel.right
                            anchors.leftMargin: spacing
                            width: 50
                            height: 30
                            checked: false
                            onCheckedChanged: {
                                if(checked){
                                    isNvidia = true;
                                }else{
                                    isNvidia = false
                                }
                            }
                        }
                     }
                 }
            }
        }
    }

    Rectangle {
        id: okRectangle
        width: parent.width * 0.5
        height: 40
        color: "#0e3e64"
        opacity: 1
        z: 1
        anchors {
            left: parent.left
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                globalSettingbox.state = "collapsed";
                setGlobalValues();
            }
        }

        Image {
            id: okImage
            x: (175 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_ok.PNG"
        }
    }

    Rectangle {
        id: cancelRectangle
        width: parent.width * 0.5
        height: 40
        color: "#1f1f1f"
        opacity: 1
        z: 1
        anchors {
            right: parent.right
            bottom: parent.bottom
        }

        MouseArea {
            anchors.fill: parent
            onClicked: {
                isYoutube = tempYouTube;
                globalSettingbox.state = "collapsed";
            }
        }

        Image {
            id: cancelImage
            x: (175 - width) / 2
            y: (parent.height - height) / 2
            width: 25
            height: 25
            fillMode: Image.PreserveAspectFit
            source: "../../resources/btn_cancel.PNG"
        }
    }

    function getGlobalValues() {
        getInputSettings();
        getFileExtension();
        getOutputSettings();
    }

    function setGlobalValues(){
        qmlMainWindow.setTempWidth(xRes.text);
        qmlMainWindow.setTempHeight(yRes.text);
        qmlMainWindow.setTempPanoWidth(xPano.text);
        qmlMainWindow.setTempPanoHeight(yPano.text);
        qmlMainWindow.setTempFPS(fps.text);
        qmlMainWindow.setTempSourceFPS(sourceFps.text);
		qmlMainWindow.setLidarPort(lidarPort.text);
		if (singleFrameRadio.checked)
		{
			qmlMainWindow.setTempStartFrame(-1);
			qmlMainWindow.setTempEndFrame(-1);
		}
		else
		{
			qmlMainWindow.setTempStartFrame(startFrame.text);
			qmlMainWindow.setTempEndFrame(endFrame.text);
		}
        qmlMainWindow.setTempCalibFile(cameraCalib.text);
        
		setFileExtension();



        // Auto Calibration Parameters
        qmlMainWindow.setTempFov(hfov.text);
        qmlMainWindow.setTempLensType(lensTypeCombo.currentIndex);
        qmlMainWindow.setQuality(51 - qualitySlider.value);
        qmlMainWindow.setTempSplitMins(splitMin.text);
        qmlMainWindow.setTempSnapshotDir(savePathFeild.text);
        //qmlMainWindow.setTempWeightMapDir(weightmapPathFeild.text);
        qmlMainWindow.enableNvidia(isNvidia);
    }

    function changeLiveMode()
    {
		settingMode = templateModes.LIVE;
        fileExtItem.visible = false;
        outItem.anchors.top = lidarPortItem.bottom;
        outItem.anchors.topMargin = 20;
    }

    function changeVideoMode()
    {
		settingMode = templateModes.VIDEO;
        fileExtItem.visible = false;
        outItem.anchors.top = lidarPortItem.bottom;
        outItem.anchors.topMargin = 20;
    }

    function changeImageMode()
    {
		settingMode = templateModes.IMAGE;
        fileExtItem.visible = true;
    }

    function clearTitle()
    {
		settingMode = templateModes.NONE;
    }

    function getFileExtension()
    {
        var fileExt = qmlMainWindow.getTempFileExt();
        if (fileExt === "dpx")
        {
            fileExtCombo.currentIndex = 0;
        }
        else if (fileExt === "tiff")
        {
            fileExtCombo.currentIndex = 1;
        }
        else if (fileExt === "jpg")
        {
            fileExtCombo.currentIndex = 2;
        }
        else if (fileExt === "png")
        {
            fileExtCombo.currentIndex = 3;
        }
        else if (fileExt === "bmp")
        {
            fileExtCombo.currentIndex = 4;
        }
    }

    function setFileExtension()
    {
        switch (fileExtCombo.currentIndex)
        {
            case 0: qmlMainWindow.setTempFileExt("dpx"); break
            case 1: qmlMainWindow.setTempFileExt("tiff"); break;
            case 2: qmlMainWindow.setTempFileExt("jpg"); break;
            case 3: qmlMainWindow.setTempFileExt("png"); break;
            case 4: qmlMainWindow.setTempFileExt("bmp"); break;
        }
    }

    function getOfflineVideoCodec()
    {
        var videoCodec = qmlMainWindow.getTempOfflineVideoCodec();
        if (videoCodec === "H.264")
        {
            rtmpVideoCombo.currentIndex = 0;
        }
        else if (videoCodec === "H.265")
        {
            rtmpVideoCombo.currentIndex = 1;
        }

    }

    function setOfflineVideoCodec()
    {
        switch (rtmpVideoCombo.currentIndex)
        {
            case 0: qmlMainWindow.setTempOfflineVideoCodec("H.264"); break
            case 1: qmlMainWindow.setTempOfflineVideoCodec("H.265"); break;
        }

    }

    function getOfflineAudioCodec()
    {
        var audioCodec = qmlMainWindow.getTempOfflineAudioCodec();
        if (!audioCodec === "AAC") return;
            rtmpAudioCombo.currentIndex = 0;
    }

    function setOfflineAudioCodec()
    {
        qmlMainWindow.setTempOfflineAudioCodec("AAC");
    }

    function getStreamVideoCodec()
    {
        var videoCodec = qmlMainWindow.getTempStreamVideoCodec();
        if (videoCodec === "H.264")
        {
            rtmpVideoCombo.currentIndex = 0;
        }
        else if (videoCodec === "H.265")
        {
            rtmpVideoCombo.currentIndex = 1;
        }
    }

    function setStreamVideoCodec()
    {
        switch (rtmpVideoCombo.currentIndex)
        {
            case 0: qmlMainWindow.setTempStreamVideoCodec("H.264"); break
            case 1: qmlMainWindow.setTempStreamVideoCodec("H.265"); break;
        }

    }

    function getStreamAudioCodec()
    {
        var audioCodec = qmlMainWindow.getTempStreamAudioCodec();
        if (!audioCodec === "AAC") return;
            rtmpAudioCombo.currentIndex = 0;
    }

    function setStreamAudioCodec()
    {
        qmlMainWindow.setTempStreamAudioCodec("AAC");
    }

    function getInputSettings()
    {
        getInputInfo();

        originPanoramaHeight = qmlMainWindow.getTempPanoHeight()
        originPanoramaWidth = qmlMainWindow.getTempPanoWidth()

        xPano.text = qmlMainWindow.getTempPanoWidth();
        yPano.text = qmlMainWindow.getTempPanoHeight();
        fps.text = qmlMainWindow.getTempFPS();
        cameraCalib.text = qmlMainWindow.getTempCalibFile();
        // Auto Calibration Parameters
        hfov.text = qmlMainWindow.getTempFov();
        lensTypeCombo.currentIndex = qmlMainWindow.getTempLensType();
    }    

    function getInputInfo()
    {
        var slotInfoList = [];
        if(settingMode === templateModes.LIVE)
        {
            slotInfoList = dshowSettingbox.slotInfoList;
            startFrame.enabled = true;
            endFrame.enabled = true;
            startFrame.text = 0;
            endFrame.text = -1;
            startFrame.enabled = false;
            endFrame.enabled = false;

            singleFrameItem.visible = false;
            frameItem.anchors.top = panoResItem.bottom;

        }
        else if(settingMode === templateModes.VIDEO)
        {
            slotInfoList = videoSettingbox.slotInfoList;
            startFrame.enabled = true;
            endFrame.enabled = true;
            startFrame.text = 0;
            endFrame.text = -1;
            startFrame.enabled = false;
            endFrame.enabled = false;

            singleFrameItem.visible = false;
            frameItem.anchors.top = panoResItem.bottom;
        }
        else if(settingMode === templateModes.IMAGE)
        {
            slotInfoList = imageSettingbox.slotInfoList;

            startFrame.enabled = true;
            endFrame.enabled = true;
            startFrame.text = -1;
            endFrame.text = -1;

            singleFrameItem.visible = true;
            frameItem.anchors.top = singleFrameItem.bottom;

            var tempStartFrame = qmlMainWindow.getTempStartFrame();
            var tempEndFrame = qmlMainWindow.getTempEndFrame();

            if(tempStartFrame === -1 && tempEndFrame === -1){
                singleFrameRadio.checked = true;
                
                startFrame.enabled = false;
                endFrame.enabled = false;
            }


            startFrame.text = root.isTemplate === true ? -1 : tempStartFrame;
            endFrame.text = root.isTemplate === true ? -1 :  tempEndFrame;
				
        }

		sourceFps.text = qmlMainWindow.getTempSourceFPS();
        if(slotInfoList.length === 3) {
            xRes.text = slotInfoList[0];
            yRes.text = slotInfoList[1];
			if (sourceFps.text === 0)
				sourceFps.text = slotInfoList[2];
        }

		if (sourceFps.text === 0)
			sourceFps.text = 30;

		lidarPort.text = qmlMainWindow.getLidarPort();
		if(lidarPort.text === -1)
			lidarPort.text = 8888;
    }

    function getOutputSettings()
    {
        refreshStreamState();
        refreshOfflineState();
        getSplitMins();
        getQuality();
        getSnapshotDir();
        getWeightmapDir();
        getNvidia();
    }

    function refreshOfflineState()
    {
        getOfflineVideoCodec();
		getOfflineAudioCodec();
    }

    function refreshStreamState()
    {
        getStreamVideoCodec();
        getStreamAudioCodec();
    }

    function getQuality()
    {
        qualitySlider.value = 51 - qmlMainWindow.getQuality();
    }

    function getSplitMins()
    {
        splitMin.text = qmlMainWindow.getTempSplitMins();
    }

    function getNvidia()
    {
        var isNvidia = false;
        isNvidia = qmlMainWindow.isNvidia();
        if(isNvidia)
            nvidiaSwitch.checked = true;
        else
            nvidiaSwitch.checked = false;
    }

    function getSnapshotDir()
    {
        savePathFeild.text = qmlMainWindow.getTempSnapshotDir();
    }

    function getWeightmapDir()
    {
        //weightmapPathFeild.text = qmlMainWindow.getTempWeightMapDir();
    }

    function setFileUrl(fileUrl){
        fileDialog.folder = fileUrl;
    }

    function getResolution(){
        xRes.text = qmlMainWindow.getTempWidth();
        yRes.text = qmlMainWindow.getTempHeight();
    }
}
