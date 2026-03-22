from __future__ import annotations

import loguru
from loguru import logger as main_logger
from datetime import datetime
from pathlib import Path

# デフォルトのハンドラ（ターミナル出力）を削除
main_logger.remove()


def setup_logger(module_name: str) -> loguru.Logger:
    """Create a module-specific logger that writes to ./logs/<module_name>_<timestamp>.log"""
    log_path = Path.cwd() / "logs" / f"{module_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    bound_logger: loguru.Logger = main_logger.bind(module=module_name)
    main_logger.add(
        log_path,
        level="DEBUG",
        filter=lambda record, m=module_name: record["extra"].get("module") == m,
    )
    return bound_logger
