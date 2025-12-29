"""
ロガー使用例.

このファイルは、ロガーモジュールの実際の使用例を示します。
"""

from __future__ import annotations

from pathlib import Path

from automl_engine.common.logger import setup_logger, shutdown_logger


def main() -> None:
    """メインの実行例."""
    # 例1: コンソールのみへのロギング
    console_logger = setup_logger("example_console")
    console_logger.info("コンソールへのログ出力")

    # 例2: ファイルとコンソールへのロギング
    log_dir = Path("./logs")
    log_file = log_dir / "example.log"
    file_logger = setup_logger("example_file", str(log_file))
    file_logger.info("ファイルとコンソールへのログ出力")

    # 例3: DEBUGレベルでのロギング
    debug_logger = setup_logger("example_debug", level="DEBUG")
    debug_logger.debug("デバッグメッセージ")
    debug_logger.info("情報メッセージ")

    # 例4: 設定ファイルを使用したロギング
    # 設定ファイルが存在する場合
    config_path = Path(__file__).parent.parent.parent / "common" / "logger" / "logger_config.ini"
    if config_path.exists():
        config_logger = setup_logger("example_config", str(log_file), str(config_path))
        config_logger.info("設定ファイルから読み込んだ設定でログ出力")

    # 例5: 例外のロギング
    exception_logger = setup_logger("example_exception")
    try:
        result = 1 / 0  # noqa: F841
    except ZeroDivisionError:
        exception_logger.exception("エラーが発生しました")

    # クリーンアップ
    shutdown_logger()


if __name__ == "__main__":
    main()
