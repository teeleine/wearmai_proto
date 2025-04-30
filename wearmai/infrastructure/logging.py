import logging
import sys

import structlog

def configure_logging():
    # =========================================================================
    # 1) stdlib logging setup
    # =========================================================================
    LOG_LEVEL = logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=LOG_LEVEL,
    )

    # Silence overly‐verbose libraries
    noisy = [
        "urllib3",        # HTTP stack
        "httpx",          # if you ever use HTTPX
        "weaviate",       # the weaviate‐client module (adjust name if needed)
    ]
    for lib in noisy:
        logging.getLogger(lib).setLevel(logging.WARNING)

    # =========================================================================
    # 2) structlog setup
    # =========================================================================
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,              # respect stdlib levels
            structlog.stdlib.add_logger_name,              # logger name
            structlog.stdlib.add_log_level,                # level
            structlog.processors.TimeStamper(fmt="iso"),   # timestamp
            structlog.processors.StackInfoRenderer(),      # stack info
            structlog.processors.format_exc_info,          # exception info
            structlog.processors.JSONRenderer(),           # JSON output
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
