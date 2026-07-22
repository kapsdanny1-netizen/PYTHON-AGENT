"""NotificationTool — console in dev/test, Slack webhook in production.

* ``ENVIRONMENT=dev|test`` → the alert is printed to stdout (and structlog);
  no network call is made. ``delivered=True`` marks the renderv path.
* ``ENVIRONMENT=prod`` → POSTs a Block-Kit style payload to
  ``SLACK_WEBHOOK_URL``; a missing webhook URL is an honest ``error`` rather
  than a silent drop.

Severity drives the visual marker (emoji + colour attachment). The interface
(channel, title, message, severity) is identical in both modes, so switching
environments requires no code change.
"""

from __future__ import annotations

from typing import ClassVar, Literal

import httpx
from pydantic import Field

from config.settings import Environment
from exceptions import ConfigurationError, ExternalServiceError
from logging_config import get_logger
from tools.base import BaseToolInput, BaseToolOutput, EnergyForgeTool

_SEVERITY_EMOJI = {"LOW": "ℹ️", "MED": "⚠️", "HIGH": "🟠", "CRITICAL": "🔴"}
_SEVERITY_COLOR = {"LOW": "#2eb886", "MED": "#daa038", "HIGH": "#e2781d", "CRITICAL": "#d00000"}

logger = get_logger("tools.notification")


class NotificationInput(BaseToolInput):
    """Input for NotificationTool."""

    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=3000)
    severity: Literal["LOW", "MED", "HIGH", "CRITICAL"] = "MED"
    asset_id: str | None = Field(default=None)
    channel: str | None = Field(default=None, description="Override channel (Slack mode)")


class NotificationOutput(BaseToolOutput):
    """Delivery result."""

    delivered: bool = False
    destination: str = ""
    severity: str = "MED"
    mode: str = "console"


class NotificationTool(EnergyForgeTool[NotificationInput, NotificationOutput]):
    """Send an operations notification.

    Console in dev/test, Slack webhook (``SLACK_WEBHOOK_URL``) in prod. Use
    for safety alerts and HITL escalation — not for routine chatter.
    """

    name: ClassVar[str] = "notification"
    description: ClassVar[str] = (
        "Send an operations notification about an asset (title, message, "
        "severity LOW/MED/HIGH/CRITICAL, optional asset_id and channel). "
        "Dev: prints to console. Prod: posts to the configured Slack webhook."
    )
    input_model: ClassVar[type[BaseToolInput]] = NotificationInput
    output_model: ClassVar[type[BaseToolOutput]] = NotificationOutput

    async def _arun(self, tool_input: NotificationInput) -> NotificationOutput:
        if self._settings.environment is Environment.PROD:
            return await self._send_slack(tool_input)
        return self._emit_console(tool_input)

    def _emit_console(self, tool_input: NotificationInput) -> NotificationOutput:
        emoji = _SEVERITY_EMOJI.get(tool_input.severity, "⚠️")
        banner = (
            f"\n{emoji} [{tool_input.severity}] {tool_input.title}\n"
            f"   asset={tool_input.asset_id or 'n/a'}\n"
            f"   {tool_input.message[:900]}\n"
        )
        print(banner)  # deliberate: dev console channel
        logger.info("notification.console", title=tool_input.title,
                    severity=tool_input.severity, asset_id=tool_input.asset_id)
        return NotificationOutput(
            delivered=True, destination="console", severity=tool_input.severity,
            mode="console", confidence=0.95,
        )

    async def _send_slack(self, tool_input: NotificationInput) -> NotificationOutput:
        webhook = self._settings.slack_webhook_url
        if webhook is None:
            raise ConfigurationError(
                "ENVIRONMENT=prod requires SLACK_WEBHOOK_URL for notifications",
                context={"env": "prod"},
            )
        channel = tool_input.channel or self._settings.notification_channel
        emoji = _SEVERITY_EMOJI.get(tool_input.severity, "⚠️")
        payload = {
            "channel": channel,
            "username": "EnergyForge Agent",
            "attachments": [{
                "color": _SEVERITY_COLOR.get(tool_input.severity, "#daa038"),
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text",
                     "text": f"{emoji} [{tool_input.severity}] {tool_input.title}"}},
                    {"type": "section", "text": {"type": "mrkdwn",
                     "text": f"*Asset:* `{tool_input.asset_id or 'n/a'}`\n{tool_input.message[:2800]}"}},
                ],
            }],
        }
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(webhook.get_secret_value(), json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(
                f"Slack webhook delivery failed: {exc}",
                context={"channel": channel},
            ) from exc
        logger.info("notification.slack", channel=channel, severity=tool_input.severity)
        return NotificationOutput(
            delivered=True, destination=f"slack:{channel}",
            severity=tool_input.severity, mode="slack", confidence=0.97,
        )
