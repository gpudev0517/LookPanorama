import QtQuick 2.0
import QtQuick.Controls.Styles 1.4
import QtQuick.Controls 1.4

SpinBox {	
	decimals: 2

	onEditingFinished: {               
		root.setFocus();
    }
}

