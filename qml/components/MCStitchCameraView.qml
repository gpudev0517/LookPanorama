import QtQuick 2.4
import MCQmlCameraView 1.0
MCStitchVideoWindow {
    id: stitchView
    destroyOnClose: true
    property alias camView : stitch
    property int stitchWidth: centralItem.width
    property int stitchHeight: centralItem.height

    MCQmlCameraView {
        id: stitch
        anchors.fill: parent
        visible: true
    }

	function closeView()
		{
			close();
		}
}

