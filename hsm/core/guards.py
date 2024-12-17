# hsm/core/guards.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import logging
from typing import Any, Callable, List

from hsm.core.errors import GuardEvaluationError
from hsm.interfaces.abc import AbstractGuard
from hsm.interfaces.protocols import Event
from hsm.runtime.async_support import AsyncGuard


class BasicGuard(AbstractGuard):
    """
    A base guard class that raises NotImplementedError.

    Runtime Invariants:
    - Calling check() will always raise NotImplementedError until overridden.

    Example:
        guard = BasicGuard()
        guard.check(event, state_data)  # Raises NotImplementedError
    """

    def check(self, event: Event, state_data: Any) -> bool:
        raise NotImplementedError("BasicGuard must be subclassed and check() overridden.")


class NoOpGuard(BasicGuard):
    """
    A guard that always returns True.

    Useful as a placeholder when no condition is required.

    Runtime Invariants:
    - Always returns True.

    Example:
        guard = NoOpGuard()
        result = guard.check(event, state_data)  # result is always True
    """

    def check(self, event: Event, state_data: Any) -> bool:
        return True


class LoggingGuard(BasicGuard):
    """
    A guard that logs the event and state_data before returning True.

    Runtime Invariants:
    - Logging is non-intrusive and does not modify event or state_data.

    Attributes:
        logger_name: Name of the logger to use.

    Example:
        guard = LoggingGuard("hsm.guard")
        result = guard.check(event, state_data)
        # Logs event ID and state_data; returns True
    """

    def __init__(self, logger_name: str = "hsm.guard"):
        self.logger = logging.getLogger(logger_name)

    def check(self, event: Event, state_data: Any) -> bool:
        event_id = event.get_id()
        self.logger.info("Guard check: event_id=%s, state_data=%s", event_id, state_data)
        return True


class KeyExistsGuard(BasicGuard):
    """
    A guard that ensures certain keys exist in state_data.
    If any required key is missing, raises GuardEvaluationError.

    Runtime Invariants:
    - State_data must be a dictionary.
    - All required keys must be present.

    Attributes:
        required_keys: The keys that must exist in state_data.

    Example:
        guard = KeyExistsGuard(["status", "counter"])
        guard.check(event, {"status": "ok", "counter": 42})  # returns True
        guard.check(event, {"status": "ok"})  # raises GuardEvaluationError
    """

    def __init__(self, required_keys: List[str]):
        self.required_keys = required_keys

    def check(self, event: Event, state_data: Any) -> bool:
        if not isinstance(state_data, dict):
            raise GuardEvaluationError(
                "state_data must be a dictionary", guard_name="KeyExistsGuard", state_data=state_data, event=event
            )

        for key in self.required_keys:
            if key not in state_data:
                raise GuardEvaluationError(
                    f"Missing required key: {key}", guard_name="KeyExistsGuard", state_data=state_data, event=event
                )

        return True


class ConditionGuard(BasicGuard):
    """
    A guard that evaluates a user-provided condition function.
    If the condition returns False, raises GuardEvaluationError.

    Runtime Invariants:
    - condition is a callable that takes state_data and returns a bool.
    - State_data must allow condition checks without errors.

    Attributes:
        condition: A callable that returns True if validation passes, False otherwise.

    Example:
        def condition(data):
            return data.get("counter", 0) > 10

        guard = ConditionGuard(condition)
        guard.check(event, {"counter": 11})  # returns True
        guard.check(event, {"counter": 5})   # raises GuardEvaluationError
    """

    def __init__(self, condition: Callable[[Any], bool]):
        self.condition = condition

    def check(self, event: Event, state_data: Any) -> bool:
        try:
            if not self.condition(state_data):
                raise GuardEvaluationError(
                    "Condition failed", guard_name="ConditionGuard", state_data=state_data, event=event
                )
        except Exception as e:
            # If condition evaluation raises an exception, wrap it in GuardEvaluationError
            raise GuardEvaluationError(
                f"Condition evaluation error: {e}", guard_name="ConditionGuard", state_data=state_data, event=event
            ) from e
        return True


class AsyncConditionGuard(BasicGuard):
    """
    A guard that checks a condition asynchronously.
    Useful if the validation requires async I/O or async computations.

    Runtime Invariants:
    - If condition fails, raises GuardEvaluationError.
    - Requires asyncio event loop for async operations.

    Attributes:
        async_condition: An async callable that returns True/False.

    Example:
        async def async_condition(data):
            await asyncio.sleep(0.01)
            return data.get("ready", False)

        guard = AsyncConditionGuard(async_condition)
        # In async code:
        # await guard.check(event, {"ready": True})  # returns True
        # await guard.check(event, {})  # raises GuardEvaluationError
    """

    def __init__(self, async_condition: Callable[[Any], Any]):
        self.async_condition = async_condition

    async def check(self, event: Event, state_data: Any) -> bool:
        try:
            result = await self.async_condition(state_data)
            if not result:
                raise GuardEvaluationError(
                    "Async condition failed", guard_name="AsyncConditionGuard", state_data=state_data, event=event
                )
        except Exception as e:
            raise GuardEvaluationError(
                f"Async condition evaluation error: {e}",
                guard_name="AsyncConditionGuard",
                state_data=state_data,
                event=event,
            ) from e
        return True


class AsyncNoOpGuard(AsyncGuard):
    """
    An async guard that always returns True.

    Useful as a placeholder when no condition is required in async state machines.

    Runtime Invariants:
    - Always returns True.
    """

    async def check(self, event: Event, state_data: Any) -> bool:
        return True
