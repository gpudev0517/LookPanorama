import QtQuick 2.4
import QmlInteractiveView 1.0

MCInteractVideoWindow {
    id: stitchView
    destroyOnClose: true
    property alias camView : interact

    QmlInteractiveView {
        id: interact
        anchors.fill: parent
        visible: true

		/*ShaderEffect {
			fragmentShader:
				"
				uniform sampler2D source;
				varying highp vec2 qt_TexCoord0;
				void main() {
					highp vec2 tc = qt_TexCoord0;
					gl_FragColor = texture2D(source, tc);
				}
				"
		}*/

		onSendClose: stitchView.close()
    }
}

