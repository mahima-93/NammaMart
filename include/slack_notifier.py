"""
include/slack_notifier.py
--------------------------
Sends a message to a Slack channel via Incoming Webhook.

Setup:
  1. Go to https://api.slack.com/apps → Create App → Incoming Webhooks → Activate
  2. Add to workspace → copy Webhook URL
  3. Set env var in Airflow:  SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
     (Admin → Variables → Add  OR  airflow_settings.yaml)
"""

import os
import json
import urllib.request


def send_slack_alert(message: str, webhook_url: str = None) -> bool:
    """
    Sends message to Slack. Returns True if successful.
    Falls back to console print if no webhook URL is configured (safe for demo).
    """
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

    if not url:
        print("[Slack] No  set. Printing alert to console instead:")
        print("=" * 60)
        print(message)
        print("=" * 60)
        return False

    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            status = resp.status
            print(f"[Slack] Alert sent successfully. HTTP {status}")
            return True
    except Exception as e:
        print(f"[Slack] Failed to send alert: {e}")
        return False
