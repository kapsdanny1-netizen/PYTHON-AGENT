"""Human-in-the-loop (HITL) approval handler.

Behaviour by environment:

* **dev**  — interactive terminal: prints the approval prompt and reads a
  decision from stdin (blocking, executed in a thread); non-interactive
  (dashboard, CI): auto-approve, loudly recorded in the audit trail.
* **test** — always auto-approve (deterministic pipelines).
* **prod** — posts the request to the Slack webhook AND pushes it onto the
  Redis list ``energyforge:hitl:requests``, then blocks on
  ``BLPOP energyforge:hitl:responses:<item_id>`` (operators answer via the
  webhook/API). Timeout = fail-safe REJECTED.
"""

from __future__ import annotations

import asyncio
import json
from enum import StrEnum
from uuid import uuid4

from config.settings import Environment, Settings, get_settings
from logging_config import get_logger
from orchestrator.state import HitlItem

logger = get_logger(__name__)


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


def make_hitl_item(reason: str, agent: str, detail: str) -> HitlItem:
    """Create a pending HITL queue entry."""
    return HitlItem(
        item_id=uuid4().hex[:12], reason=reason, agent=agent,
        detail=detail, status="pending",
    )


async def request_approval(
    item: HitlItem, *, trace_id: str, settings: Settings | None = None
) -> ApprovalDecision:
    """Route an approval request per environment (see module docstring)."""
    cfg = settings or get_settings()
    if cfg.environment is Environment.PROD:
        return await _prod_approval(item, trace_id=trace_id, settings=cfg)
    if cfg.environment is Environment.TEST:
        logger.info("hitl.auto_approve_test", reason=item["reason"])
        return ApprovalDecision.AUTO_APPROVED
    import sys

    if sys.stdin.isatty():
        return await _tty_approval(item)
    logger.warning("hitl.auto_approve_non_interactive", reason=item["reason"])
    return ApprovalDecision.AUTO_APPROVED


async def _tty_approval(item: HitlItem) -> ApprovalDecision:
    """Dev path: block on stdin (threaded to keep the loop free)."""
    print(
        f"\n┌─ HITL APPROVAL REQUIRED ({item['item_id']}) ──────────────"
        f"\n│ reason : {item['reason']}"
        f"\n│ agent  : {item['agent']}"
        f"\n│ detail : {item['detail'][:400]}"
        f"\n└─ approve? [y/N] ",
    )
    answer = await asyncio.to_thread(input, "> ")
    decision = ApprovalDecision.APPROVED if answer.strip().lower() in ("y", "yes") else ApprovalDecision.REJECTED
    logger.info("hitl.decision_tty", decision=decision.value, item_id=item["item_id"])
    return decision


async def _prod_approval(item: HitlItem, *, trace_id: str, settings: Settings) -> ApprovalDecision:
    """Prod path: Slack notify + Redis queue read (webhook answers BLPUSH)."""
    import redis.asyncio as aioredis

    client = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await client.rpush("energyforge:hitl:requests", json.dumps(dict(item)))
        if settings.slack_webhook_url is not None:
            from tools.notification import NotificationInput, NotificationTool

            await NotificationTool(settings).run(NotificationInput(
                title="HITL approval required",
                message=(f"*Reason:* {item['reason']}\n*Agent:* {item['agent']}\n"
                         f"*Detail:* {item['detail'][:500]}\n"
                         f"Reply via the approval webhook with item `{item['item_id']}`."),
                severity="HIGH",
            ))
        pending = await client.blpop(
            f"energyforge:hitl:responses:{item['item_id']}", timeout=300
        )
        if pending is None:
            logger.warning("hitl.timeout_rejected", item_id=item["item_id"])
            return ApprovalDecision.REJECTED  # fail-safe
        verdict = str(pending[1]).strip().lower()
        decision = ApprovalDecision.APPROVED if verdict in ("approve", "approved", "yes") else ApprovalDecision.REJECTED
        logger.info("hitl.decision_prod", decision=decision.value, item_id=item["item_id"])
        return decision
    finally:
        await client.aclose()
