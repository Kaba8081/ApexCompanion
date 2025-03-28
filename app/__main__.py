
import sys
import traceback
from loguru import logger
from typing import Type
from types import TracebackType

from app.utils import AppInfo
from app.views.dialogue import show_fatal_error
#from app.controllers import AppController

def handle_exception(
    exc_type: Type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None,
) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    else:
        logger.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback),
        )

        show_fatal_error(
            title="Apex Companion crashed",
            text = "An unhandled exception occurred. Please report this issue on GitHub.",
            information = f"Version: {AppInfo.VERSION}",
            details = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
        )

    sys.exit()

if __name__ == "__main__":
    pass