#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: bash deploy/upgrade.sh <patchX.Y.Z.zip 路径>"
  echo "示例: bash ~/trading-python/deploy/upgrade.sh ~/patch5.0.3.zip"
  exit 1
fi

ZIP_PATH="$(realpath "$1")"
if [[ ! -f "$ZIP_PATH" ]]; then
  echo "❌ 找不到补丁包: $1"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKUP_DIR="$HOME/trading_backups"
STAMP="$(date +%Y%m%d_%H%M%S)"

cd "$APP_DIR"
mkdir -p "$BACKUP_DIR" data

PY_BIN="$APP_DIR/.venv/bin/python"
if [[ ! -x "$PY_BIN" ]]; then
  PY_BIN="$(command -v python3)"
fi

echo "📦 项目目录: $APP_DIR"
echo "📦 补丁文件: $ZIP_PATH"

# 1. 检测运行方式并优雅停止（确保 SQLite WAL 完整刷盘）
RUN_MODE="nohup"
if command -v systemctl >/dev/null 2>&1 && systemctl --user list-unit-files trading-advisor.service >/dev/null 2>&1; then
  if systemctl --user is-enabled --quiet trading-advisor.service 2>/dev/null || systemctl --user is-active --quiet trading-advisor.service 2>/dev/null; then
    RUN_MODE="systemd"
  fi
fi

if [[ "$RUN_MODE" == "systemd" ]]; then
  echo "🛑 停止 systemd 服务 trading-advisor.service..."
  systemctl --user stop trading-advisor.service || true
fi

if pgrep -f "python.*-m src\.main" >/dev/null 2>&1; then
  echo "🛑 优雅停止后台进程 (SIGTERM)..."
  pkill -SIGTERM -f "python.*-m src\.main" || true
  for _ in {1..10}; do
    if ! pgrep -f "python.*-m src\.main" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if pgrep -f "python.*-m src\.main" >/dev/null 2>&1; then
    echo "⚠️ 进程未在 10 秒内退出，强制终止..."
    pkill -SIGKILL -f "python.*-m src\.main" || true
    sleep 1
  fi
fi

# 2. 备份数据库与配置（仅保留最近 5 份，主库始终完整保留）
DB_BAK=""
CFG_BAK=""
if [[ -f "$APP_DIR/data/advisor.db" ]]; then
  DB_BAK="$BACKUP_DIR/advisor.db.bak.$STAMP"
  cp -a "$APP_DIR/data/advisor.db" "$DB_BAK"
  echo "💾 已备份数据库: $DB_BAK"
  ls -1t "$BACKUP_DIR"/advisor.db.bak.* 2>/dev/null | tail -n +6 | xargs -r rm -f
fi

if [[ -f "$APP_DIR/config.yaml" ]]; then
  CFG_BAK="$BACKUP_DIR/config.yaml.bak.$STAMP"
  cp -a "$APP_DIR/config.yaml" "$CFG_BAK"
  echo "💾 已备份配置: $CFG_BAK"
  ls -1t "$BACKUP_DIR"/config.yaml.bak.* 2>/dev/null | tail -n +6 | xargs -r rm -f
fi

# 3. 覆盖解压代码
echo "📂 解压更新文件..."
"$PY_BIN" -m zipfile -e "$ZIP_PATH" "$APP_DIR"

# 4. 检查自定义配置差异提示
if [[ -n "$CFG_BAK" ]] && ! cmp -s "$CFG_BAK" "$APP_DIR/config.yaml"; then
  echo "ℹ️ 提示: config.yaml 已更新。若你此前手动改过 watchlist/universe，可执行以下命令核对:"
  echo "   diff -u \"$CFG_BAK\" \"$APP_DIR/config.yaml\""
fi

# 5. 编译与离线单元测试验收（失败则中止，不启动异常版本）
echo "🧪 运行编译与单元测试..."
"$PY_BIN" -m compileall -q src tests
"$PY_BIN" -m tests.run_tests tests

# 6. 启动后台服务
if [[ "$RUN_MODE" == "systemd" ]]; then
  echo "🚀 启动 systemd 服务 trading-advisor.service..."
  systemctl --user start trading-advisor.service
  sleep 2
  systemctl --user --no-pager -l status trading-advisor.service | head -n 12 || true
else
  echo "🚀 使用 nohup 重新启动后台进程..."
  nohup "$PY_BIN" -m src.main > "$APP_DIR/bot.log" 2>&1 < /dev/null &
  sleep 2
  if pgrep -af "python.*-m src\.main"; then
    echo "✅ 进程已就绪，最近日志:"
    tail -n 10 "$APP_DIR/bot.log" || true
  else
    echo "❌ 启动失败，请检查 $APP_DIR/bot.log"
    exit 1
  fi
fi

# 7. 清理本次补丁包
rm -f "$ZIP_PATH"
echo "🎉 升级完成并已清理补丁包 $ZIP_PATH"
