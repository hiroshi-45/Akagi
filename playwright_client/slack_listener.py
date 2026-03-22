import os
import queue
import time
import threading
from dataclasses import dataclass
from dotenv import load_dotenv
from slack_bolt import App
import os, glob
from typing import Optional
from slack_bolt.adapter.socket_mode import SocketModeHandler

ATTACH_LATEST_IMAGE = os.getenv("ATTACH_LATEST_IMAGE", "true").lower() in {"1","true","yes","on"}

CLICK_PROOF_DIR = os.getenv(
    "CLICK_PROOF_DIR",
    "/Users/maruno/source/Akagi/logs/click_proof"
)

def _find_latest_image(dir_path: str) -> Optional[str]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp")
    paths = []
    for pat in exts:
        paths.extend(glob.glob(os.path.join(dir_path, pat)))
    if not paths:
        return None
    # 更新時刻で最新を取得
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]

# 外部から見るフラグ
stop_event = threading.Event()    # 「対局終了」
logout_event = threading.Event()  # 「ログアウト」

# 受付・完了を同スレッドに返信するための文脈
@dataclass
class ThreadContext:
    channel: str
    thread_ts: str

_stop_ctx_lock   = threading.Lock()
_logout_ctx_lock = threading.Lock()
_stop_ctx: ThreadContext | None = None
_logout_ctx: ThreadContext | None = None

def _set_stop_ctx(ctx: ThreadContext):
    global _stop_ctx
    with _stop_ctx_lock:
        _stop_ctx = ctx

def pop_stop_ctx() -> ThreadContext | None:
    global _stop_ctx
    with _stop_ctx_lock:
        ctx, _stop_ctx = _stop_ctx, None
        return ctx

def _set_logout_ctx(ctx: ThreadContext):
    global _logout_ctx
    with _logout_ctx_lock:
        _logout_ctx = ctx

def pop_logout_ctx() -> ThreadContext | None:
    global _logout_ctx
    with _logout_ctx_lock:
        ctx, _logout_ctx = _logout_ctx, None
        return ctx

BOT_TOKEN  = os.getenv("SLACK_BOT_TOKEN", "")

APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
TARGET_CHANNEL = os.getenv("TARGET_CHANNEL", "C09BT8ZHYTE")
STOP_KEYWORD    = os.getenv("STOP_KEYWORD", "終了")
LOGOUT_KEYWORD = os.getenv("STOP_KEYWORD", "ログアウト")
AUTHCODE_KEYWORD = os.getenv("AUTHCODE_KEYWORD", "認証コード")
LOGIN_KEYWORD = os.getenv("LOGIN_KEYWORD", "ログイン")

app = App(token=BOT_TOKEN)

@dataclass
class AuthcodeItem:
    channel: str
    thread_ts: str
    text: str

_authcode_queue: "queue.Queue[AuthcodeItem]" = queue.Queue()
_wait_auth_lock = threading.Lock()
# root_ts -> (channel, start_ts)
_wait_auth: dict[str, tuple[str, float]] = {}

def pop_authcode_item_nowait() -> AuthcodeItem | None:
    try:
        return _authcode_queue.get_nowait()
    except queue.Empty:
        return None

# def post_in_thread(channel: str, thread_ts: str, text: str):
#     app.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)

def post_in_thread(channel: str, thread_ts: str, text: str, attach_latest=False):
    """
    同スレッド返信。attach_latest=True なら CLICK_PROOF_DIR の最新画像を添付。
    添付できなければ本文のみを送信。
    """
    if attach_latest and ATTACH_LATEST_IMAGE:
        latest = _find_latest_image(CLICK_PROOF_DIR)
        if latest and os.path.exists(latest):
            # files_upload_v2 は thread_ts に対応（Slack SDK v3.26+）
            try:
                app.client.files_upload_v2(
                    channel=channel,
                    thread_ts=thread_ts,
                    initial_comment=text,
                    file=latest,
                    filename=os.path.basename(latest),
                    title=os.path.basename(latest),
                )
                return
            except Exception as e:
                # 失敗したら本文のみでフォールバック
                app.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=f"{text}\n(画像添付失敗: {e})")
                return

    # 画像添付しない/見つからない場合
    app.client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=text)

# 追加: 「次の投稿を待っているスレッド」を記録
_wait_login_lock = threading.Lock()
# root_ts -> (channel, start_ts)
_wait_login: dict[str, tuple[str, float]] = {}

# Playwright側へ渡すログインアイテム
@dataclass
class LoginItem:
    channel: str
    thread_ts: str
    text: str

_login_queue: "queue.Queue[LoginItem]" = queue.Queue()

def pop_login_item_nowait() -> LoginItem | None:
    try:
        return _login_queue.get_nowait()
    except queue.Empty:
        return None

@app.event("message")
def handle_message(event, logger):
    if event.get("subtype") == "bot_message":
        return
    if event.get("channel") != TARGET_CHANNEL:
        return

    text = (event.get("text") or "").strip()
    if not text:
        return

    root_ts = event.get("thread_ts") or event.get("ts")
    channel = event["channel"]

    # 1) 「認証コード」受付
    if AUTHCODE_KEYWORD in text:
        with _wait_auth_lock:
            _wait_auth[root_ts] = (channel, time.time())
        post_in_thread(channel, root_ts, "🔏『認証コード』を受け付けました。このスレッドの**次のメッセージ**でコードを送ってください。")
        return

    # 2) 受付済みスレッドの「次の投稿」を回収
    with _wait_auth_lock:
        if root_ts in _wait_auth:
            ch, _ = _wait_auth.pop(root_ts)
            item = AuthcodeItem(channel=ch, thread_ts=root_ts, text=text)
            _authcode_queue.put(item)
            # エコーバックは先頭40文字だけ
            post_in_thread(ch, root_ts, f"🧾 認証コードを受け取りました：`{text[:40] + ('…' if len(text) > 40 else '')}`\n処理を開始します…")
            return


    if LOGIN_KEYWORD in text:
        with _wait_login_lock:
            _wait_login[root_ts] = (channel, time.time())
        post_in_thread(channel, root_ts, "🔑『ログイン』を受け付けました。このスレッドに**次のメッセージ**で文字列を送ってください（それを使って処理します）。")
        return

    # --- 2) 受付済みスレッドの「次の投稿」を回収 ---
    with _wait_login_lock:
        if root_ts in _wait_login:
            ch, _ = _wait_login.pop(root_ts)
            # 文字列をキューへ
            raw_text = (event.get("text") or "").strip()
            clean_text = normalize_slack_text(raw_text)

            item = LoginItem(channel=ch, thread_ts=root_ts, text=clean_text)
            _login_queue.put(item)
            post_in_thread(ch, root_ts, f"📝 文字列を受け取りました：`{clean_text[:40] + ('…' if len(clean_text) > 40 else '')}`\n処理を開始します…")

            return

    if STOP_KEYWORD in text:
        post_in_thread(channel, root_ts, "🛑『対局終了』を受け付けました。後処理を実行します…")
        _set_stop_ctx(ThreadContext(channel=channel, thread_ts=root_ts))
        logger.info(f"[slack_listener] STOP_KEYWORD detected: {text}")
        stop_event.set()

    if LOGOUT_KEYWORD in text:
        post_in_thread(channel, root_ts, "🔐『ログアウト』を受け付けました。ログアウトします…")
        _set_logout_ctx(ThreadContext(channel=channel, thread_ts=root_ts))
        logger.info(f"[slack_listener] LOGOUT_KEYWORD detected: {text}")
        logout_event.set()

def start_socket_mode_in_thread():
    if not (BOT_TOKEN and APP_TOKEN and TARGET_CHANNEL):
        print("[slack_listener] SLACK_BOT_TOKEN / SLACK_APP_TOKEN / TARGET_CHANNEL が未設定のため、Slack連携は無効化されます。", flush=True)
        return None
    th = threading.Thread(
        target=lambda: SocketModeHandler(app, APP_TOKEN).start(),
        name="SlackSocketMode",
        daemon=True,
    )
    th.start()
    return th

import re
import html
import unicodedata

MAILTO_RE = re.compile(r"<mailto:([^>|]+)\|([^>]+)>")
LINK_RE   = re.compile(r"<([^>|]+)\|([^>]+)>")
ANGLE_RE  = re.compile(r"^<([^>]+)>$")  # 例: <mailto:foo@bar>

def normalize_slack_text(raw: str) -> str:
    if not raw:
        return ""
    s = raw

    # 1) <mailto:addr|label> → label（labelが無ければaddrでも可）
    s = MAILTO_RE.sub(lambda m: m.group(2) or m.group(1), s)

    # 2) <url|label> → label（URLリンク化のとき）
    s = LINK_RE.sub(lambda m: m.group(2) or m.group(1), s)

    # 3) 角括弧だけで包まれたもの（<mailto:addr> など）→ 中身
    m = ANGLE_RE.match(s)
    if m:
        s = m.group(1)

    # 4) HTMLエンティティ解除（&lt; &gt; &amp; など）
    s = html.unescape(s)

    # 5) 全角半角のゆれをNFKCで正規化（任意）
    s = unicodedata.normalize("NFKC", s).strip()
    return s
