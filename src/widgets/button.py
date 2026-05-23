"""Styled push-buttons for the SmogNet application."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton

_STYLE = """
QPushButton {
    background-color: #cba6f7;
    color: #1e1e2e;
    border: 2px solid #585b70;
    border-radius: 6px;
    padding: 8px 14px;
    font-weight: bold;
    font-size: 13px;
}
QPushButton:hover {
    background-color: #d0bcfe;
}
QPushButton:pressed {
    background-color: #b4befe;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #585b70;
}
"""


class SmogButton(QPushButton):
    """A QPushButton styled to the SmogNet colour scheme.

    Args:
        text: Label shown on the button.
        parent: Optional parent widget.
    """

    def __init__(self, text: str, parent=None) -> None:
        """Creates the button and applies SmogNet styling.

        Args:
            text: Label shown on the button.
            parent: Optional parent widget.
        """
        super().__init__(text, parent)
        self.setStyleSheet(_STYLE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
