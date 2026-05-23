"""Styled text labels for the SmogNet application."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class SmogLabel(QLabel):
    """A QLabel pre-styled to the SmogNet colour scheme.

    Args:
        text: Initial label text.
        parent: Optional parent widget.
    """

    def __init__(self, text: str = "", parent=None) -> None:
        """Creates the label with SmogNet text styling.

        Args:
            text: Initial label text.
            parent: Optional parent widget.
        """
        super().__init__(text, parent)
        self.setStyleSheet(
            "color: #cdd6f4; font-size: 13px; background: transparent;"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
