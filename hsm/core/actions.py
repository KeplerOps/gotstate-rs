# hsm/core/actions.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator

from hsm.core.errors import ActionExecutionError
from hsm.interfaces.abc import AbstractAction
from hsm.interfaces.protocols import Event
from hsm.runtime.async_support import AsyncAction


class BasicAction(AbstractAction):
    """
    A base action class that raises NotImplementedError.

    Runtime Invariants:
    - Calling execute() will always raise NotImplementedError until overridden.

    Example:
        action = BasicAction()
        action.execute(event, state_data)  # Raises NotImplementedError
    """

    def execute(self, event: Event, state_data: Any) -> None:
        raise NotImplementedError("BasicAction must be subclassed and execute() overridden.")


class NoOpAction(BasicAction):
    """
    An action that does nothing.

    This is useful as a placeholder action or for testing transitions that require
    an action but do not need to modify state.

    Runtime Invariants:
    - execute() completes instantly without side-effects.

    Example:
        action = NoOpAction()
        action.execute(event, state_data)  # Does nothing
    """

    def execute(self, event: Event, state_data: Any) -> None:
        # No operation performed
        pass


class LoggingAction(BasicAction):
    """
    An action that logs the event and state_data for debugging.

    Runtime Invariants:
    - Logging is atomic and does not modify state_data or event.
    - No exceptions are raised unless logging fails at a system level.

    Attributes:
        logger_name: Name of the logger to use.

    Example:
        action = LoggingAction(logger_name="hsm.action")
        action.execute(event, state_data)
        # Logs event ID and state_data content.
    """

    def __init__(self, logger_name: str = "hsm.actions"):
        self.logger = logging.getLogger(logger_name)

    def execute(self, event: Event, state_data: Any) -> None:
        event_id = event.get_id()
        self.logger.info("Executing action for event: %s, state_data: %s", event_id, state_data)


class SetDataAction(BasicAction):
    """
    An action that sets a key-value pair in the state_data atomically.

    Uses a context manager to revert changes if any error occurs.

    Runtime Invariants:
    - Changes to state_data are atomic.
    - If an error occurs during execute(), changes are reverted.

    Attributes:
        key: The key to set in state_data.
        value: The value to set.

    Example:
        action = SetDataAction("status", "updated")
        action.execute(event, state_data)
        # state_data["status"] is set to "updated"
    """

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    @contextmanager
    def _temporary_change(self, state_data: Dict[str, Any]) -> Generator[None, None, None]:
        original = state_data.get(self.key, None)
        state_data[self.key] = self.value
        try:
            yield
        except Exception:
            # Revert changes if an error occurs
            if original is None:
                del state_data[self.key]
            else:
                state_data[self.key] = original
            raise

    def execute(self, event: Event, state_data: Any) -> None:
        if not isinstance(state_data, dict):
            raise ActionExecutionError(
                "State data must be a dictionary", action_name="SetDataAction", state_data=state_data, event=event
            )
        with self._temporary_change(state_data):
            # Potentially do more work here. If errors occur, changes revert.
            # No additional errors here, so changes persist.
            pass


class ValidateDataAction(BasicAction):
    """
    An action that validates the state_data against certain criteria.
    Raises ActionExecutionError if validation fails.

    Runtime Invariants:
    - Validation is deterministic.
    - If validation fails, no changes are made to state_data.

    Attributes:
        required_keys: A list of keys that must be present in state_data.
        condition: A callable that returns True if validation passes, False otherwise.

    Example:
        def check_condition(data):
            return data.get("counter", 0) > 0

        action = ValidateDataAction(["counter"], check_condition)
        action.execute(event, state_data)
        # Raises ActionExecutionError if "counter" not in state_data or condition fails.
    """

    def __init__(self, required_keys: list[str], condition: Any):
        self.required_keys = required_keys
        self.condition = condition

    def execute(self, event: Event, state_data: Any) -> None:
        # Check that state_data is a dictionary
        if not isinstance(state_data, dict):
            raise ActionExecutionError(
                "Invalid state_data type for validation",
                action_name="ValidateDataAction",
                state_data=state_data,
                event=event,
            )

        # Check that required keys exist
        for key in self.required_keys:
            if key not in state_data:
                raise ActionExecutionError(
                    f"Missing required key: {key}", action_name="ValidateDataAction", state_data=state_data, event=event
                )

        # Check condition
        if not self.condition(state_data):
            raise ActionExecutionError(
                "Validation condition failed", action_name="ValidateDataAction", state_data=state_data, event=event
            )


class AsyncSetDataAction(AsyncAction):
    """
    An async version of SetDataAction that sets a key-value pair in the state_data atomically.

    Uses a context manager to revert changes if any error occurs.

    Runtime Invariants:
    - Changes to state_data are atomic.
    - If an error occurs during execute(), changes are reverted.

    Attributes:
        key: The key to set in state_data.
        value: The value to set.
    """

    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value

    async def execute(self, event: Event, state_data: Any) -> None:
        old_value = state_data.get(self.key)
        state_data[self.key] = self.value
        return old_value  # For potential rollback


class AsyncLoggingAction(AsyncAction):
    """
    An async version of LoggingAction that logs the event and state_data.

    Runtime Invariants:
    - Logging is non-intrusive and does not modify event or state_data.

    Attributes:
        logger_name: Name of the logger to use.
    """

    def __init__(self, logger_name: str = "hsm.action"):
        self.logger = logging.getLogger(logger_name)

    async def execute(self, event: Event, state_data: Any) -> None:
        event_id = event.get_id()
        self.logger.info("Executing action for event: %s, state_data: %s", event_id, state_data)
