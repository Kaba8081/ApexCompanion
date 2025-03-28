import os
import sys
from pathlib import Path
from typing import Tuple

from loguru import logger
from PySide6.QtCore import QEvent, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from app.utils import AppInfo

def show_fatal_error(
    title: str = "Fatal error",
    text: str = "A fatal error occured!",
    information: str = "Please report the error on GitHub.",
    details: str = "",
) -> None:
    logger.info(
        f"Showing fatal error box with input: [{title}], [{text}], [{information}], [{details}]"
    )
    diag = FatalErrorDialog(title, text, information, details)
    diag.exec_()
    return

def _setup_error_icon(
        diag: QDialog,
        details_btn: QPushButton | None = None,
) -> QVBoxLayout:
    l_layout = QVBoxLayout()
    piximap = getattr(QStyle, "SP_MessageBoxCritical")
    icon = diag.style().standardIcon(piximap)
    label = QLabel()
    label.setPixmap(icon.pixmap(64, 64))
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    l_layout.addWidget(label)
    if details_btn is not None:
        l_layout.addWidget(details_btn)
    l_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    return l_layout

def FatalErrorDialog(_BaseDialogue):
    """Custom error to display fatal errors."""

    def __init__(
            self,
            title: str = "Fatal error",
            text: str = "A fatal error occured!",
            information: str = "Please report the error on GitHub.",
            details: str = "",
            parent: QWidget | None = None,
        ) -> None:
        super().__init__(parent)

        self.text = text
        self.information = information
        self.details = details

        self.details_btn = QPushButton("Show Details")
        self.close_btn = QPushButton("Close")
        self.open_log_btn = QPushButton("Open Log")

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.open_log_btn)
        btn_layout.addWidget(self.close_btn)

        # Details
        self.details_edit = QPlainTextEdit()
        self.details_edit.setPlainText(self.details)
        self.details_edit.setMaximumHeight(150)
        self.details_edit.setReadOnly(True)
        self.details_edit.setHidden(True)

        # Layout
        layout = QVBoxLayout()
        main_layout = QVBoxLayout()
        main_layout.setAlignment(Qt.AligmentFlag.AlignLeft)

        # Left-side
        l_layout = _setup_error_icon(self, self.details_btn)
        main_layout.addLayout(l_layout)

        # Center spacer
        main_layout.addItem(QSpacerItem(20, 20))

        # Right-side
        r_layout = QVBoxLayout()
        
        txt = QLabel(self.text)
        txt.setWordWrap(True)
        r_layout.addWidget(txt)

        info = QLabel(self.information)
        info.setWordWrap(True)
        r_layout.addWidget(info)

        r_layout.addLayout(btn_layout)
        main_layout.addLayout(r_layout)

        layout.addLayout(main_layout)
        layout.addWidget(self.details_edit)

        self.setLayout(layout)
        self.setFixedWitdth(self.sizeHint().width())

        def _toggle_details() -> None:
            self.details_edit.setHidden(not self.details_edit.isHidden())
            if self.details_edit.isHidden():
                self.details_btn.setText("Show Details")
            else:
                self.details_btn.setText("Hide Details")
            self.adjustSize()

        self.close_btn.clicked.connect(self.close)
        # TODO : check all os.startfile calls
        self.open_log_btn.clicked.connect(
            lambda: os.startfile(AppInfo().app_log_file)
        )