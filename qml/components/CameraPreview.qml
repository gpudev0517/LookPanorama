import QtQuick 2.4
import MCQmlCameraView 1.0
MCStitchVideoWindow {
    id: stitchView
    destroyOnClose: true
    property alias camView : preview

    MCQmlCameraView {
        id: preview
        anchors.fill: parent
        visible: true
    }

    function closeView()
        {
            close();
        }
}

