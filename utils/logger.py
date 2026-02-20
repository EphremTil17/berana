import logging
import sys
from typing import ClassVar

# Define constants for log categorizations
DEBUG = "DEBUG"
INFO = "INFO"
SUCCESS = "SUCCESS"
WARNING = "WARNING"
ERROR = "ERROR"

# Add a custom SUCCESS level to the logging library
SUCCESS_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS_LEVEL_NUM, SUCCESS)


def success(self, message: str, *args, **kws) -> None:
    """Custom log level for successful operations.

    Args:
        self: Logger instance.
        message: Success message.
        *args: Variable arguments.
        **kws: Keyword arguments.
    """
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        # Yes, logger takes its '*args' as 'args'.
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)


logging.Logger.success = success


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


def get_logger(name="TranslateGemma"):
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

    return logger


# Example usage:
if __name__ == "__main__":
    log = get_logger()
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.success("This is a success message!")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
