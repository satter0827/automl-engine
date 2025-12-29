"""
ロガーモジュールのテスト.
"""

from __future__ import annotations

import configparser
import logging
import tempfile
from pathlib import Path

import pytest

from automl_engine.common.logger import get_logger, setup_logger, shutdown_logger


class TestLoggerSetup:
    """ロガーのセットアップに関するテスト."""

    def teardown_method(self) -> None:
        """各テスト後にロガーをシャットダウン."""
        shutdown_logger()

    def test_setup_logger_console_only(self) -> None:
        """コンソール出力のみのロガーをセットアップできる."""
        logger = setup_logger("test_console")

        assert logger is not None
        assert logger.name == "test_console"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)

    def test_setup_logger_with_file(self) -> None:
        """ファイル出力を含むロガーをセットアップできる."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test.log"
            logger = setup_logger("test_file", str(log_path))

            assert logger is not None
            assert len(logger.handlers) == 2  # コンソール + ファイル
            assert log_path.exists()

            # ログを書き込んでファイルに出力されることを確認
            logger.info("Test message")
            assert log_path.read_text().find("Test message") != -1

    def test_setup_logger_custom_level(self) -> None:
        """カスタムログレベルでロガーをセットアップできる."""
        logger = setup_logger("test_level", level="DEBUG")

        assert logger.level == logging.DEBUG

        logger2 = setup_logger("test_level2", level="ERROR")
        assert logger2.level == logging.ERROR

    def test_setup_logger_creates_directory(self) -> None:
        """ログファイルのディレクトリが存在しない場合は自動作成される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "logs" / "app" / "test.log"
            logger = setup_logger("test_mkdir", str(log_path))

            assert log_path.parent.exists()
            assert log_path.exists()
            logger.info("Test")

    def test_setup_logger_with_config_file(self) -> None:
        """設定ファイルからロガーをセットアップできる."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 設定ファイルを作成
            config_path = Path(tmpdir) / "test_config.ini"
            config = configparser.RawConfigParser()
            config["logger"] = {
                "level": "WARNING",
                "format": "%(levelname)s - %(message)s",
                "date_format": "%H:%M:%S",
            }
            with open(config_path, "w", encoding="utf-8") as f:
                config.write(f)

            # 設定ファイルを使ってロガーをセットアップ
            logger = setup_logger("test_config", config_file=str(config_path))

            assert logger.level == logging.WARNING

    def test_setup_logger_config_file_not_found(self) -> None:
        """存在しない設定ファイルを指定するとエラーになる."""
        with pytest.raises(FileNotFoundError):
            setup_logger("test_not_found", config_file="/nonexistent/config.ini")

    def test_get_logger_existing(self) -> None:
        """既存のロガーを取得できる."""
        logger1 = setup_logger("test_get")
        logger2 = get_logger("test_get")

        assert logger1 is logger2

    def test_get_logger_nonexistent(self) -> None:
        """存在しないロガーを取得するとNoneが返る."""
        logger = get_logger("nonexistent")
        assert logger is None

    def test_setup_logger_reuse_existing(self) -> None:
        """同じ名前でsetup_loggerを呼ぶと既存のロガーを再利用する."""
        logger1 = setup_logger("test_reuse")
        logger2 = setup_logger("test_reuse")

        assert logger1 is logger2

    def test_shutdown_logger(self) -> None:
        """ロガーをシャットダウンできる."""
        setup_logger("test_shutdown")
        assert get_logger("test_shutdown") is not None

        shutdown_logger()
        assert get_logger("test_shutdown") is None


class TestLoggerOutput:
    """ロガーの出力に関するテスト."""

    def teardown_method(self) -> None:
        """各テスト後にロガーをシャットダウン."""
        shutdown_logger()

    def test_logger_writes_to_file(self) -> None:
        """ロガーがファイルに正しく書き込む."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "output.log"
            logger = setup_logger("test_write", str(log_path))

            # 各レベルのログを書き込む
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")

            # ファイルの内容を確認
            content = log_path.read_text()
            # デフォルトはINFOレベルなのでDEBUGは出力されない
            assert "Debug message" not in content
            assert "Info message" in content
            assert "Warning message" in content
            assert "Error message" in content
            assert "Critical message" in content

    def test_logger_format(self) -> None:
        """ログのフォーマットが正しい."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "format.log"
            logger = setup_logger("test_format", str(log_path))

            logger.info("Format test")

            content = log_path.read_text()
            # フォーマットが含まれているか確認
            assert "[INFO]" in content
            assert "test_format" in content
            assert "Format test" in content

    def test_logger_multiple_messages(self) -> None:
        """複数のメッセージを連続して書き込める."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "multiple.log"
            logger = setup_logger("test_multiple", str(log_path))

            for i in range(10):
                logger.info(f"Message {i}")

            content = log_path.read_text()
            for i in range(10):
                assert f"Message {i}" in content

    def test_logger_with_exception(self) -> None:
        """例外情報を含むログを書き込める."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "exception.log"
            logger = setup_logger("test_exception", str(log_path))

            try:
                raise ValueError("Test exception")
            except ValueError:
                logger.exception("An error occurred")

            content = log_path.read_text()
            assert "An error occurred" in content
            assert "ValueError" in content
            assert "Test exception" in content


class TestLoggerConfiguration:
    """ロガーの設定に関するテスト."""

    def teardown_method(self) -> None:
        """各テスト後にロガーをシャットダウン."""
        shutdown_logger()

    def test_config_rotation_settings(self) -> None:
        """設定ファイルでローテーション設定を指定できる."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "rotation_config.ini"
            log_path = Path(tmpdir) / "rotation.log"

            # ローテーション設定を含む設定ファイルを作成
            config = configparser.RawConfigParser()
            config["logger"] = {
                "level": "INFO",
                "rotation_when": "midnight",
                "rotation_interval": "1",
                "backup_count": "7",
            }
            with open(config_path, "w", encoding="utf-8") as f:
                config.write(f)

            logger = setup_logger("test_rotation", str(log_path), str(config_path))
            logger.info("Test rotation")

            # ログファイルが作成されていることを確認
            assert log_path.exists()

    def test_config_custom_format(self) -> None:
        """設定ファイルでカスタムフォーマットを指定できる."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "format_config.ini"
            log_path = Path(tmpdir) / "custom_format.log"

            # カスタムフォーマットの設定ファイルを作成
            config = configparser.RawConfigParser()
            config["logger"] = {
                "level": "INFO",
                "format": "%(levelname)s | %(message)s",
            }
            with open(config_path, "w", encoding="utf-8") as f:
                config.write(f)

            logger = setup_logger("test_custom_format", str(log_path), str(config_path))
            logger.info("Custom format test")

            content = log_path.read_text()
            assert "INFO | Custom format test" in content

    def test_level_argument_overrides_config(self) -> None:
        """引数で指定したレベルが設定ファイルより優先される."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "level_config.ini"

            # WARNING レベルの設定ファイル
            config = configparser.RawConfigParser()
            config["logger"] = {"level": "WARNING"}
            with open(config_path, "w", encoding="utf-8") as f:
                config.write(f)

            # 引数で DEBUG を指定
            logger = setup_logger("test_level_override", config_file=str(config_path), level="DEBUG")

            # 引数が優先されることを確認
            assert logger.level == logging.DEBUG
