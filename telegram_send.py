import time
import threading
import logging
import requests

from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

log = logging.getLogger("security_system")

_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def _send_photo(img_path: str, caption: str) -> bool:
    """POST image + caption to the configured Telegram chat."""
    url = f"{_BASE_URL}/sendPhoto"
    try:
        with open(img_path, "rb") as photo:
            response = requests.post(
                url,
                data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                files={"photo": photo},
                timeout=10,
            )
        if response.ok:
            return True
        log.error("Telegram API error: %s %s", response.status_code, response.text)
        return False
    except Exception:
        log.exception("Exception while sending photo to Telegram.")
        return False


def _send_message(text: str) -> bool:
    """POST a plain-text message to the configured Telegram chat (fallback)."""
    url = f"{_BASE_URL}/sendMessage"
    try:
        response = requests.post(
            url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=10,
        )
        if response.ok:
            return True
        log.error("Telegram API error: %s %s", response.status_code, response.text)
        return False
    except Exception:
        log.exception("Exception while sending message to Telegram.")
        return False


def _telegram_send(msg: str, img_path: str) -> bool:
    """Send alert with image, retrying up to 3 times, then falling back to text."""
    for attempt in range(1, 4):
        try:
            if _send_photo(img_path, caption=msg):
                log.info("Telegram alert sent successfully (attempt %d).", attempt)
                return True
        except Exception:
            log.exception("Telegram send failed, attempt %d", attempt)
        time.sleep(2)

    # Final fallback: text-only if all photo attempts failed
    log.warning("All photo attempts failed; sending text-only alert.")
    return _send_message(msg)


def send_telegram_async(msg: str, img_path: str):
    """Fire-and-forget Telegram alert in a background daemon thread."""
    threading.Thread(
        target=_telegram_send,
        args=(msg, img_path),
        daemon=True,
    ).start()