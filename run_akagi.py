from pathlib import Path
from akagi.logging_utils import setup_logger
from akagi.akagi import main

# logs フォルダが無ければ作成
Path("logs").mkdir(exist_ok=True)

logger = setup_logger("akagi_main")

if __name__ == "__main__":
    logger.info("=== Akagi 起動 ===")
    main()
