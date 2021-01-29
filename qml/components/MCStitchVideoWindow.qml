import QtQuick 2.4
import "../controls"

FloatingStitchWindow  {
    property int uiSpacing: 4
    property bool active
    property bool destroyOnClose: false
    property  int  toolwindowHandleHeight: 0
    id: root
    width: 350
    height: 350
    windowMenuGroup: 'video'
    maximizable: true
    minimizable: true
    signal closing()
}

