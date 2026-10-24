STYLE = """
QDialog {
    background: transparent;
    color: #fff7fb;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QFrame#surface {
    background-color: #21171d;
    border: 1px solid #63384e;
    border-radius: 9px;
}
QLabel#eyebrow {
    color: #f18abb;
    font-size: 8pt;
    font-weight: 700;
}
QLabel#title {
    color: #fff7fb;
    font-size: 19pt;
    font-weight: 600;
}
QLabel#section {
    color: #c9aebb;
    font-size: 8pt;
    font-weight: 700;
}
QLabel#fieldLabel {
    color: #eadce4;
}
QLabel#value {
    color: #ff9ccc;
    font-weight: 600;
}
QLabel#status {
    color: #b39eaa;
    font-size: 8pt;
}
QLabel#fpsValue {
    color: #ff9ccc;
    font-size: 28pt;
    font-weight: 700;
}
QLabel#fpsUnit {
    color: #c9aebb;
    font-size: 10pt;
}
QComboBox, QDoubleSpinBox {
    background: #30232b;
    color: #ffffff;
    border: 1px solid #634254;
    border-radius: 6px;
    min-height: 34px;
    padding: 0 10px;
    selection-background-color: #a94375;
}
QDoubleSpinBox QLineEdit {
    background: transparent;
    color: #ffffff;
    selection-background-color: #a94375;
}
QComboBox:hover, QDoubleSpinBox:hover {
    border-color: #f18abb;
}
QComboBox::drop-down {
    border: 0;
    width: 28px;
}
QComboBox QAbstractItemView {
    background: #30232b;
    color: #ffffff;
    border: 1px solid #634254;
    selection-background-color: #a94375;
    padding: 4px;
}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 18px;
    border: 0;
}
QFrame#divider {
    background: #49323e;
    max-height: 1px;
}
QPushButton#mode {
    background: #30232b;
    border: 1px solid #634254;
    border-radius: 6px;
    color: #d1bec9;
    min-height: 36px;
    padding: 0 8px;
}
QPushButton#mode:hover {
    background: #402a37;
    color: #ffffff;
}
QPushButton#mode:checked {
    background: #8e3c67;
    border-color: #f18abb;
    color: #ffffff;
}
QToolButton#windowControl, QToolButton#closeControl {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    color: #f3cada;
    font-size: 14pt;
    min-width: 32px;
    max-width: 32px;
    min-height: 30px;
    max-height: 30px;
}
QToolButton#windowControl:hover, QToolButton#closeControl:hover {
    background: #402a37;
    border-color: #70445c;
    color: #ffffff;
}
QToolButton#closeControl:hover {
    background: #b63c6e;
    border-color: #ef79a8;
}
"""
