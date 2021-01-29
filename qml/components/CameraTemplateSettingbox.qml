import QtQuick 2.5
import QtQuick.Window 2.2
import QtQuick.Controls.Styles.Flat 1.0 as Flat
import QtQuick.Extras 1.4
import QtQuick.Extras.Private 1.0
import QtQuick.Controls 1.4
import QtQuick.Controls.Styles 1.4
import QtQuick.Dialogs 1.2
import "../controls"

ToolWindow{
    id: details
    width: 350
    height: 550
    z: 10
    windowState: qsTr("windowed")
    visible: true

    property string   title: "Camera Template"
    property int      fontSize: 15
    property int      itemHeight: 30
    property int      spacing: 20
    property color    textColor: "#ffffff"
    property int      lblWidth: 60

	property int        leftMargin: 20
	property int        rightMargin: 0	
	property int        lblFont: 14
	property var        opacityVal: 0.9
	property int        camWidth: 100
	property int        camHeight: 100    
    property int         m_curCameraIndex;
	property bool       m_isReady: false

	Item {
        id: cameraParamsItem
		width : 330
        height: parent.height

		property bool       isHovered : false		

		FileDialog {
            id: loadFileDialog
            title: "Load PAS file"
            selectMultiple: false
            nameFilters: [ "PAS file (*.pas)"]
            selectedNameFilter: "All files (*)"
            onAccepted: {
				var imagePath = fileUrl.toString().substring(8); // Remove "file:///" prefix
				qmlMainWindow.loadTemplatePAS(cameraCombo.currentIndex, imagePath);
				getCameraParams(cameraCombo.currentIndex);
                setCameraParams()
            }
            onRejected: {
            }
        }

        FileDialog {
            id: saveFileDialog
            title: "Save PAS file"
            selectExisting: false
            selectFolder: false
            selectMultiple: false
            nameFilters: [ "PAS file (*.pas)"]
            selectedNameFilter: "All files (*)"
            onAccepted: {
				var imagePath = fileUrl.toString().substring(8); // Remove "file:///" prefix
				qmlMainWindow.saveTemplatePAS(cameraCombo.currentIndex, imagePath);
            }
            onRejected: {
            }
        }

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
				anchors.topMargin: spacing * 1.6			

                Item {
                    id: cameraItem
                    width: parent.width
                    height: 30
                    anchors.top: parent.top
                    anchors.topMargin: 10

                    Text {
                        id: cameraText
                        color: textColor
                        width: lblWidth
                        text: qsTr("Camera")
                        horizontalAlignment: Text.AlignLeft
                        anchors.left: parent.left
                        anchors.leftMargin: spacing
                        anchors.verticalCenter: parent.verticalCenter
                        font.pixelSize: 13
                    }

                    ComboBox {
                        id: cameraCombo
                        width: parent.width / 2
                        height: 30
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.left: cameraText.right
                        anchors.leftMargin: spacing
                        model: ListModel {
                            id: camListModel
                        }

                        onCurrentTextChanged: {
                            getCameraParams(currentIndex);
                        }
                    }
                }

				 Item {
                    id: lensTypeItem
                    width: parent.width
                    height: 30
                    anchors.top: cameraItem.bottom
                    anchors.topMargin: spacing

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
                            getCameraParams(m_curCameraIndex);
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
                    anchors.top: lensTypeItem.bottom
					anchors.topMargin: spacing

					Slider {
						id: fovSlider
                        value: fovSpin.value
                        width: parent.width * 0.7 - spacing * 2
						minimumValue: 0.1
						maximumValue: 360
						stepSize: 0.01
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
                                cameraParamsItem.setCameraParams();
							}
						}
					}

                    FlatSpin {
                        id: fovSpin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: fovSlider.maximumValue
                        minimumValue: fovSlider.minimumValue
                        stepSize: fovSlider.stepSize
                        value: fovSlider.value
                        onValueChanged: {
                            fovSlider.value = fovSpin.value;
                            cameraParamsItem.setCameraParams();
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
                        value: fovYSpin.value
                        width: parent.width * 0.7 - leftMargin * 2
						minimumValue: 0.1
						maximumValue: 360
						stepSize: 0.01
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
								setCameraParams();
							}
						}
					}

                    FlatSpin {
                        id: fovYSpin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: fovYSlider.maximumValue
                        minimumValue: fovYSlider.minimumValue
                        stepSize: fovYSlider.stepSize
                        value: fovYSlider.value
                        onValueChanged: {
                            fovYSlider.value = fovYSpin.value;
                            cameraParamsItem.setCameraParams();
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
                        value: k1Spin.value
                        width: parent.width * 0.7 - spacing * 2
						minimumValue: -2
						maximumValue: 2
						stepSize: 0.01
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
                                cameraParamsItem.setCameraParams();
							}
						}

					}

                    FlatSpin {
                        id: k1Spin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: k1Slider.maximumValue
                        minimumValue: k1Slider.minimumValue
                        stepSize: k1Slider.stepSize
                        value: k1Slider.value
                        onValueChanged: {
                            k1Slider.value = k1Spin.value;
                            cameraParamsItem.setCameraParams();
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
                        value: k2Spin.value
                        width: parent.width * 0.7 - spacing * 2
						minimumValue: -2
						maximumValue: 2
						stepSize: 0.001
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
                                cameraParamsItem.setCameraParams();
							}
						}
					}

                    FlatSpin {
                        id: k2Spin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: k2Slider.maximumValue
                        minimumValue: k2Slider.minimumValue
                        decimals: 3
                        stepSize: k2Slider.stepSize
                        value: k2Slider.value
                        onValueChanged: {
                            k2Slider.value = k2Spin.value;
                            cameraParamsItem.setCameraParams();
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
                        value: k3Spin.value
                        width: parent.width * 0.7 - spacing * 2
						minimumValue: -2
						maximumValue: 2
						stepSize: 0.0001
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
                                cameraParamsItem.setCameraParams();
							}
						}

					}

                    FlatSpin {
                        id: k3Spin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: k3Slider.maximumValue
                        minimumValue: k3Slider.minimumValue
                        decimals: 4
                        stepSize: k3Slider.stepSize
                        value: k3Slider.value
                        onValueChanged: {
                            k3Slider.value = k3Spin.value;
                            cameraParamsItem.setCameraParams();
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
                        value: offsetXSpin.value
                        width: parent.width * 0.7 - spacing * 2
						minimumValue: -camWidth / 2
						maximumValue: camWidth / 2
						stepSize: 0.1
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
                                cameraParamsItem.setCameraParams();
							}
						}

					}

                    FlatSpin {
                        id: offsetXSpin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: offsetXSlider.maximumValue
                        minimumValue: offsetXSlider.minimumValue
                        stepSize: offsetXSlider.stepSize
                        value: offsetXSlider.value
                        onValueChanged: {
                            offsetXSlider.value = offsetXSpin.value;
                            cameraParamsItem.setCameraParams();
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
                        value: offsetYSpin.value
                        width: parent.width * 0.7 - spacing * 2
						minimumValue: -camHeight / 2
						maximumValue: camHeight/ 2
						stepSize: 0.1
						anchors.left: parent.left
						anchors.leftMargin: leftMargin
						anchors.verticalCenter: parent.verticalCenter
						updateValueWhileDragging: true
						onPressedChanged: {
							if (pressed) {
                                cameraParamsItem.setCameraParams();
							}
						}
					}

                    FlatSpin {
                        id: offsetYSpin
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin
                        maximumValue: offsetYSlider.maximumValue
                        minimumValue: offsetYSlider.minimumValue
                        stepSize: offsetYSlider.stepSize
                        value: offsetYSlider.value
                        onValueChanged: {
                            offsetYSlider.value = offsetYSpin.value;
                            cameraParamsItem.setCameraParams();
                        }
                    }
				}

                Item {
                    id: loadAndSaveItem
                    height: 30
                    width: 330
                    anchors.top: offsetYItem.bottom
                    anchors.topMargin: spacing * 2

                    Item {
                        id: loadItem
                        anchors.right: saveItem.left
                        anchors.rightMargin: 20
                        width: 65
                        height: 30

                        Rectangle {
                            id: loadHoverRect
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
                            id: loadText
                            z: 1
                            color: "#ffffff"
                            text: "Load"
                            font.pointSize: 11
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            anchors.fill: parent
                        }

                        Rectangle {
                            id: loadRect
                            width: parent.width
                            height: parent.height
                           anchors.fill: parent
                            color: "#373737"

                            MouseArea {
                                id: loadMouseArea
                                width: 60
                                anchors.fill: parent
                                hoverEnabled: true

                                onEntered: loadHoverRect.visible = true
                                onExited: loadHoverRect.visible = false

                                onClicked: {
									loadFileDialog.open();
                                }
                            }
                        }
                    }

                    Item {
                        id: saveItem
                        width: 65
                        height: 30
                        anchors.right: parent.right
                        anchors.rightMargin: rightMargin

                        Rectangle {
                            id: saveHoverRect
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
                            id: saveText
                            z: 1
                            color: "#ffffff"
                            text: "Save"
                            font.pointSize: 11
                            horizontalAlignment: Text.AlignHCenter
                            verticalAlignment: Text.AlignVCenter
                            anchors.fill: parent
                        }

                        Rectangle {
                            id: saveRect
                            width: parent.width
                            height: parent.height
                           anchors.fill: parent
                            color: "#373737"

                            MouseArea {
                                id: saveMouseArea
                                width: 60
                                anchors.fill: parent
                                hoverEnabled: true

                                onEntered: saveHoverRect.visible = true
                                onExited: saveHoverRect.visible = false

                                onClicked: {
									saveFileDialog.open();
                                }
                            }
                        }
                    }
                }
			}
		}

		function getCurLensType()
		{
			return qmlMainWindow.getLensType(cameraCombo.currentIndex);
		}

        function getCameraParams(curCameraIndex){
			if(!qmlMainWindow.start) return;

			m_isReady = false;

			// check lensType
			var lensType = qmlMainWindow.getLensType(curCameraIndex);
            camWidth = qmlMainWindow.getWidth(curCameraIndex);
            camHeight = qmlMainWindow.getHeight(curCameraIndex)
            fovSlider.value = qmlMainWindow.getFov(curCameraIndex);
			fovYSlider.value = qmlMainWindow.getFovy(curCameraIndex);
            k1Slider.value = qmlMainWindow.getK1(curCameraIndex);
            k2Slider.value = qmlMainWindow.getK2(curCameraIndex);
            k3Slider.value = qmlMainWindow.getK3(curCameraIndex);
            offsetXSlider.value = qmlMainWindow.getOffsetX(curCameraIndex);
            offsetYSlider.value = qmlMainWindow.getOffsetY(curCameraIndex);

            lensTypeCombo.currentIndex = lensType;
            if (lensType === 3) {
				// LensType_opencvLens_Standard
				fovLabel.text = "FoV";				
				fovYLabel.visible = false;
				fovYItem.visible = false;						 
				k1Item.anchors.top = fovItem.bottom;
			} else if (lensType == 4) {
				// LensType_opencvLens_Fisheye
				fovLabel.text = "FoVX";				
				fovYLabel.text = "FoVY";				
				fovYLabel.visible = true;
				fovYItem.visible = true;			
				k1Item.anchors.top = fovYItem.bottom;
			} else {
				fovLabel.text = "FoV";				
				fovYLabel.visible = false;
				fovYItem.visible = false;			 
				k1Item.anchors.top = fovItem.bottom;
			}

			m_isReady = true;
        }

        function setCameraParams(){

            if(!qmlMainWindow.start || !m_isReady) return;

            qmlMainWindow.setFov(fovSlider.value, m_curCameraIndex);
			qmlMainWindow.setFovy(fovYSlider.value, m_curCameraIndex);
            qmlMainWindow.setK1(k1Slider.value, m_curCameraIndex);
            qmlMainWindow.setK2(k2Slider.value, m_curCameraIndex);
            qmlMainWindow.setK3(k3Slider.value, m_curCameraIndex);
            qmlMainWindow.setOffsetX(offsetXSlider.value, m_curCameraIndex);
            qmlMainWindow.setOffsetY(offsetYSlider.value, m_curCameraIndex);
            qmlMainWindow.reStitch(true);
        }
   }

    function appendCameraCombo(){
        camListModel.clear();

        for(var i = 0; i < qmlMainWindow.getCameraCount(); i++){
            camListModel.append({"text": "Camera" + (i + 1)})
        }

        cameraCombo.currentIndex = 0;
     }

	function getCameraParams(curCameraIndex){
        m_curCameraIndex = curCameraIndex;
		cameraParamsItem.getCameraParams(curCameraIndex);
	}

    function setCameraParams(){
        cameraParamsItem.setCameraParams();
	}

    function initCameraParms(){
        appendCameraCombo();
        getCameraParams(cameraCombo.currentIndex);
    }
}
