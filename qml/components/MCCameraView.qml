import QtQuick 2.4
import MCQmlCameraView 1.0
import "./"
MCVideoWindow {
    id: cameraView
    width: 400
    height: 330
    destroyOnClose: true
    property alias camView : display
    property alias drawView: background
    property int deviceNum: 0
    property var cameraIndex: if (display) display.cameraNumber
    property int name
	property int referenceCount: 0
	property bool active: false;
    property int isWindowed: 0
    property int win_x: 0
    property int win_y: 0
    property int win_rotate: 0
    property int win_z: 0
    property int win_width: 0
    property int win_height: 0
    property int win_camNum: 0

    caption:
	{
		if (display)
			qsTr(display.cameraViewName)
		else
			""
	}

	onActiveChanged: {
		if(active)
			qmlMainWindow.updateCameraView(camView,deviceNum);
		else
			qmlMainWindow.disableCameraView(camView);
	}

    MCQmlCameraView {
        id: display
        anchors.fill: parent
        visible: true

        function closeView()
        {
            close();
        }

        MouseArea {
            id: resizeMouseArea
            width: 5
            height: 5
            x: parent.width - width
            y: parent.height - height

            property bool isMoveMode: false
            property var clickPos: "1,1"
            property real movedX: 0
            property real movedY: 0

            anchors.fill: parent;
            acceptedButtons: Qt.LeftButton | Qt.RightButton

            onClicked: {

            }
            onDoubleClicked: {
                if (mouse.button == Qt.LeftButton)
                {
                    //toggleMaximizing()
                    liveArea.liveCam_index = cameraIndex;
                    expandWindow()
                }
            }
        }

        onSendClose :
		{
			cameraView.close()
		}
    }

    Rectangle {
        id: background
        anchors.fill: parent
        opacity: 0.0
    }

    LiveCameraSetting {
        id: liveCameraSetting
        anchors.right: parent.right
        anchors.rightMargin: 30
        width: 250
        height: 0
        z: 1
        state: "collapsed"

           states: [
               State {
                   name: "collapsed"
                   PropertyChanges { target: liveCameraSetting; height: 0}
                   PropertyChanges { target:  liveCameraSetting;width: 0

                   }
               },
               State {
                   name: "expanded"
                   PropertyChanges { target: liveCameraSetting; height: 300}
                   PropertyChanges {target: liveCameraSetting;width: 200}
               }
           ]

           transitions: [
               Transition {
                   NumberAnimation { target: liveCameraSetting; property: "height"; duration: 300 }
                   NumberAnimation { target: liveCameraSetting;property: "width";duration: 300}
               }
           ]

    }

    function setCameraSetting()
    {
        if(liveCameraSetting.state == "expanded"){
            liveCameraSetting.state = "collapsed";

        }else if(liveCameraSetting.state == "collapsed"){
            liveCameraSetting.state = "expanded";
            liveCameraSetting.getCameraValues()
        }
    }

    function expandWindow() {

            if (isWindowed == 0) {

                win_x = cameraView.x
                win_y = cameraView.y
                win_z = cameraView.z
                win_width = cameraView.width
                win_height = cameraView.height

                cameraView.x = 0
                cameraView.y = 0
                cameraView.z = 2
                cameraView.width = centralItem.width
                cameraView.height = centralItem.height
                isWindowed = 1
                liveArea.setMaximizedLiveCamView();
                liveArea.resetLiveCamViewSize();

            }else if (isWindowed == 1) {

                cameraView.x = win_x
                cameraView.y = win_y
                cameraView.z = win_z
                cameraView.rotation = win_rotate;
                cameraView.width = win_width;
                cameraView.height = win_height;
                isWindowed = 0
                liveArea.setLiveCamViewSize();
            }
    }

}
