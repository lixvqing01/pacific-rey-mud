
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
from loguru import logger
import sys

class StreamlitSink:
    """
    A loguru sink that writes log messages to Streamlit's session state.
    Log messages are stored in st.session_state.log_messages.
    """
    def write(self, message):
        """
        Writes the log record to the Streamlit session state.
        Initializes st.session_state.log_messages if it doesn't exist.
        """
        log_record = message.record
        log_entry = {
            "time": log_record["time"].strftime("%Y-%m-%d %H:%M:%S"),
            "level": log_record["level"].name,
            "message": log_record["message"],
        }
        if "log_messages" not in st.session_state:
            st.session_state.log_messages = []
        st.session_state.log_messages.append(log_entry)

    def get_logs(self):
        """
        Retrieves all log messages stored in the Streamlit session state.
        Returns an empty list if no logs are present.
        """
        if "log_messages" not in st.session_state:
            return []
        return st.session_state.log_messages

    def clear_logs(self):
        """
        Clears all log messages from the Streamlit session state.
        """
        if "log_messages" in st.session_state:
            st.session_state.log_messages = []


def add_file_sink(log_file_path: str, level: str = "DEBUG"):
    """
    Adds a file sink to the global loguru logger.

    Args:
        log_file_path (str): The path to the log file.
        level (str, optional): The logging level for this sink. Defaults to "DEBUG".
    """
    logger.add(
        log_file_path,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,
        backtrace=True,
        diagnose=True
    )
    logger.info(f"File sink added for: {log_file_path} with level {level}")


logger.remove()


try:
    if get_script_run_ctx(suppress_warning=True) is not None:
        logger.add(StreamlitSink(), level="INFO", format="{message}")
        logger.info("StreamlitSink added to logger.")
    else:

        logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
        logger.info("Streamlit session_state not available. Added stderr sink as fallback for StreamlitSink.")
except (ImportError, AttributeError) as e:

    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    logger.warning(f"Streamlit not available or session_state missing ({e}). Added stderr sink instead of StreamlitSink.")


logger.add(
    sys.stderr,
    level="DEBUG",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=True
)
logger.info("Default stderr sink added to logger.")


_streamlit_sink_instance = None

def _initialize_streamlit_sink():
    global _streamlit_sink_instance
    if _streamlit_sink_instance is None:
        _streamlit_sink_instance = StreamlitSink()
    return _streamlit_sink_instance

def get_streamlit_sink() -> StreamlitSink:
    """
    Returns the global instance of StreamlitSink.
    Initializes it if it hasn't been already.
    """
    return _initialize_streamlit_sink()


_active_file_sinks = {}

def setup_logger(run_specific_log_dir: str = None, level: str = "DEBUG"):
    """
    Configures the global loguru logger idempotently.

    This function clears all existing handlers and sets up new ones based on
    the execution context (Streamlit or standard Python) and provided arguments.
    This ensures no duplicate handlers are created across multiple calls.

    Args:
        run_specific_log_dir (str, optional): Path to a directory for run-specific logs.
        level (str, optional): The logging level for file and console sinks.
    """
    logger.remove()


    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,
        diagnose=True
    )


    try:
        if get_script_run_ctx(suppress_warning=True) is not None:
            sink_instance = get_streamlit_sink()
            logger.add(sink_instance, level="INFO", format="{message}")
            logger.debug("Streamlit sink configured.")
    except (ImportError, AttributeError, RuntimeError):
        logger.debug("Streamlit context not available, skipping Streamlit sink.")


    if run_specific_log_dir:
        from pathlib import Path
        log_dir_path = Path(run_specific_log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir_path / "run_log.txt"

        logger.add(
            str(log_file_path),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
        logger.info(f"Run-specific file sink configured at: {log_file_path}")


setup_logger()


