# ロガーモジュール

標準的かつ安全な汎用ロガー基盤を提供します。

## 特徴

- **コンソールとファイルへの同時出力**: 両方への出力をサポート
- **日次ローテーション**: `TimedRotatingFileHandler`による自動ログファイルローテーション
- **設定ファイルベースの柔軟な設定**: INI形式の設定ファイルで簡単にカスタマイズ可能
- **型安全**: 完全な型アノテーション
- **Googleスタイルのdocstring**: 明確で読みやすいドキュメント

## 基本的な使い方

### コンソールのみへのロギング

```python
from automl_engine.common.logger import setup_logger

logger = setup_logger("my_module")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")
```

### ファイルへのロギング

```python
from automl_engine.common.logger import setup_logger

logger = setup_logger("my_module", "/path/to/logfile.log")
logger.info("このメッセージはファイルとコンソールの両方に出力されます")
```

### カスタムログレベル

```python
from automl_engine.common.logger import setup_logger

# DEBUGレベルのログも出力
logger = setup_logger("my_module", level="DEBUG")
logger.debug("This is a debug message")
```

### 設定ファイルを使用

```python
from automl_engine.common.logger import setup_logger

logger = setup_logger(
    "my_module",
    log_file_path="/path/to/logfile.log",
    config_file="/path/to/logger_config.ini"
)
logger.info("設定ファイルから設定を読み込みました")
```

### 例外のロギング

```python
from automl_engine.common.logger import setup_logger

logger = setup_logger("my_module")

try:
    result = 10 / 0
except ZeroDivisionError:
    logger.exception("エラーが発生しました")
```

## 設定ファイル

設定ファイルは INI 形式で記述します。サンプルとして `logger_config.ini.example` を提供しています。

### 設定ファイルのセットアップ

1. サンプルファイルをコピー:
   ```bash
   cp logger_config.ini.example logger_config.ini
   ```

2. 必要に応じて設定を編集

3. アプリケーションで使用:
   ```python
   logger = setup_logger("app", "/var/log/app.log", "logger_config.ini")
   ```

### 設定項目

```ini
[logger]
# ログレベル: DEBUG, INFO, WARNING, ERROR, CRITICAL
level = INFO

# ログフォーマット
format = %(asctime)s [%(levelname)s] %(name)s: %(message)s

# 日時フォーマット
date_format = %Y-%m-%d %H:%M:%S

# ローテーション設定
rotation_when = midnight    # ローテーションのタイミング
rotation_interval = 1       # ローテーション間隔
backup_count = 0           # 保持するバックアップファイル数（0=無制限）
```

### ローテーション設定

`rotation_when` で指定できる値:

- `S`: 秒ごと
- `M`: 分ごと
- `H`: 時間ごと
- `D`: 日ごと
- `midnight`: 深夜0時（推奨）
- `W0`～`W6`: 曜日ごと（0=月曜日）

## API リファレンス

### `setup_logger(name, log_file_path=None, config_file=None, level=None)`

ロガーをセットアップして返します。

**引数:**
- `name` (str): ロガー名（通常はモジュール名: `__name__`）
- `log_file_path` (str | None): ログファイルの出力パス（Noneの場合はコンソールのみ）
- `config_file` (str | None): 設定ファイルのパス（INI形式）
- `level` (str | None): ログレベル（"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"）

**戻り値:**
- `logging.Logger`: 設定済みのロガーインスタンス

### `get_logger(name)`

既に設定済みのロガーを取得します。

**引数:**
- `name` (str): ロガー名

**戻り値:**
- `logging.Logger | None`: ロガーインスタンス、存在しない場合はNone

### `shutdown_logger()`

全てのロガーとハンドラをシャットダウンします。

アプリケーション終了時に呼び出すことで、適切にリソースを解放します。

## セキュリティに関する注意事項

- **設定ファイルには機密情報を含めないでください**
- `logger_config.ini` は `.gitignore` に登録されているため、実際の設定ファイルはGit管理外です
- パスワードやAPIキーなどの機密情報はログに出力しないよう注意してください
- 本番環境では適切なファイルパーミッションを設定してください

## ベストプラクティス

1. **モジュールごとにロガーを作成**
   ```python
   logger = setup_logger(__name__)
   ```

2. **適切なログレベルを使用**
   - `DEBUG`: 詳細なデバッグ情報
   - `INFO`: 一般的な情報
   - `WARNING`: 警告（問題ではないが注意が必要）
   - `ERROR`: エラー（機能の一部が動作しない）
   - `CRITICAL`: 重大なエラー（アプリケーション全体に影響）

3. **例外は `logger.exception()` を使用**
   ```python
   try:
       # 処理
   except Exception:
       logger.exception("予期しないエラーが発生しました")
   ```

4. **ファイルパスは絶対パスを推奨**
   ```python
   from pathlib import Path
   log_path = Path("/var/log/myapp/app.log")
   logger = setup_logger(__name__, str(log_path))
   ```

5. **アプリケーション終了時にシャットダウン**
   ```python
   from automl_engine.common.logger import shutdown_logger
   
   try:
       # アプリケーションのメイン処理
   finally:
       shutdown_logger()
   ```

## テスト

テストを実行するには:

```bash
pytest tests/test_logger.py -v
```

## ライセンス

このプロジェクトのライセンスに従います。
