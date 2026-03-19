import logging
import sys
from typing import Any, ClassVar

# Define constants for log categorizations
DEBUG = "DEBUG"
INFO = "INFO"
SUCCESS = "SUCCESS"
WARNING = "WARNING"
ERROR = "ERROR"

# Add a custom SUCCESS level to the logging library
SUCCESS_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL_NUM, SUCCESS)


class SuccessLogger:
    """Typed wrapper around ``logging.Logger`` with a success helper."""

    def __init__(self, logger: logging.Logger) -> None:
        """Wrap a standard logger instance."""
        self._logger = logger

    def success(self, message: str, *args: Any, **kws: Any) -> None:
        """Emit one success-level log message."""
        if self._logger.isEnabledFor(SUCCESS_LEVEL_NUM):
            self._logger.log(SUCCESS_LEVEL_NUM, message, *args, **kws)

    def debug(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Delegate debug logging."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Delegate info logging."""
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Delegate warning logging."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Delegate error logging."""
        self._logger.error(msg, *args, **kwargs)

    def exception(self, msg: object, *args: Any, **kwargs: Any) -> None:
        """Delegate exception logging."""
        self._logger.exception(msg, *args, **kwargs)

    @property
    def handlers(self) -> list[logging.Handler]:
        """Expose underlying handlers for initialization checks."""
        return self._logger.handlers

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the underlying logger."""
        return getattr(self._logger, name)


# ANSI Color Codes (Simplest standard colors, no bold)
class Colors:
    """ANSI color codes for premium terminal output."""

    DEBUG = "\033[0;36m"  # Cyan
    INFO = "\033[0;34m"  # Blue
    SUCCESS = "\033[0;32m"  # Green
    WARNING = "\033[0;33m"  # Yellow
    ERROR = "\033[0;31m"  # Red
    RESET = "\033[0m"


class CustomFormatter(logging.Formatter):
    """Vibrant color-coded formatting for a premium terminal experience.

    Format: [CATEGORY] | Message.
    """

    # Pre-compute formatters
    LEVEL_FORMATTERS: ClassVar[dict[int, logging.Formatter]] = {
        logging.DEBUG: logging.Formatter(f"{Colors.DEBUG}[{DEBUG}]{Colors.RESET} | %(message)s"),
        logging.INFO: logging.Formatter(f"{Colors.INFO}[{INFO}]{Colors.RESET} | %(message)s"),
        SUCCESS_LEVEL_NUM: logging.Formatter(
            f"{Colors.SUCCESS}[{SUCCESS}]{Colors.RESET} | %(message)s"
        ),
        logging.WARNING: logging.Formatter(
            f"{Colors.WARNING}[{WARNING}]{Colors.RESET} | %(message)s"
        ),
        logging.ERROR: logging.Formatter(f"{Colors.ERROR}[{ERROR}]{Colors.RESET} | %(message)s"),
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with categorical colors and handle multi-line inputs.

        Args:
            record: The logging record to format.

        Returns:
            The formatted log string.
        """
        formatter = self.LEVEL_FORMATTERS.get(
            record.levelno, logging.Formatter("[%(levelname)s] | %(message)s")
        )

        if isinstance(record.msg, str) and "\n" in record.msg:
            lines = record.msg.split("\n")
            formatted_lines = []
            for line in lines:
                # Create a temporary record for each line
                temp_record = logging.LogRecord(
                    record.name,
                    record.levelno,
                    record.pathname,
                    record.lineno,
                    line,
                    record.args,
                    record.exc_info,
                    record.funcName,
                )
                formatted_lines.append(formatter.format(temp_record))
            return "\n".join(formatted_lines)

        return formatter.format(record)


def get_logger(name: str = "TranslateGemma") -> SuccessLogger:
    """Returns a modular logger instance."""
    logger = logging.getLogger(name)

    # Only add handlers if they haven't been added already
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # Create console handler and set level to debug
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)

        # Add custom formatter to ch
        ch.setFormatter(CustomFormatter())

        # Add ch to logger
        logger.addHandler(ch)

    return SuccessLogger(logger)


# Example usage:
if __name__ == "__main__":
    log = get_logger()
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.success("This is a success message!")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
