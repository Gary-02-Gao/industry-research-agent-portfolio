"""Credential-blind authorization gate evaluated immediately before tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ActionGateRequest(BaseModel):
    """Only trusted intent, permissions, and the proposed action are accepted."""

    model_config = ConfigDict(extra="forbid")

    user_intent: str = Field(min_length=1, max_length=4000)
    permissions: frozenset[str]
    tool_name: str = Field(min_length=1, max_length=80)
    params: dict[str, Any]


class ActionGateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowed: bool
    reason_code: str
    required_permission: str | None
    tool_name: str


_TOOL_PERMISSIONS = {
    "web_search": "search:web",
    "knowledge_search": "read:knowledge",
    "text2sql": "read:database",
    "data_analyzer": "compute:local",
    "chart_generator": "compute:local",
    "stock_query": "read:market",
    "bidding_search": "read:bidding",
    "finish": None,
}
_FORBIDDEN_PARAM_KEYS = frozenset({
    "api_key", "token", "password", "secret", "credential", "private_key",
    "system_prompt", "developer_message",
})


def evaluate_action(request: ActionGateRequest) -> ActionGateDecision:
    required = _TOOL_PERMISSIONS.get(request.tool_name)
    if request.tool_name not in _TOOL_PERMISSIONS:
        return ActionGateDecision(
            allowed=False, reason_code="unknown_tool", required_permission=None,
            tool_name=request.tool_name,
        )
    forbidden = {str(key).lower() for key in request.params} & _FORBIDDEN_PARAM_KEYS
    if forbidden:
        return ActionGateDecision(
            allowed=False, reason_code="credential_parameter_forbidden",
            required_permission=required, tool_name=request.tool_name,
        )
    if required and required not in request.permissions:
        return ActionGateDecision(
            allowed=False, reason_code="permission_denied",
            required_permission=required, tool_name=request.tool_name,
        )
    return ActionGateDecision(
        allowed=True, reason_code="authorized", required_permission=required,
        tool_name=request.tool_name,
    )


def permissions_from_context(metadata: dict[str, Any]) -> frozenset[str]:
    explicit = metadata.get("action_permissions")
    if isinstance(explicit, (list, tuple, set, frozenset)):
        return frozenset(str(item) for item in explicit)
    permissions = {"compute:local"}
    if metadata.get("search_web", True):
        permissions.update({"search:web", "read:market", "read:bidding"})
    if metadata.get("search_local", True):
        permissions.add("read:knowledge")
    if metadata.get("database_access", False):
        permissions.add("read:database")
    return frozenset(permissions)


__all__ = [
    "ActionGateDecision", "ActionGateRequest", "evaluate_action",
    "permissions_from_context",
]
