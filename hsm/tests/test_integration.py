# hsm/tests/test_integration.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

import pytest

from hsm.core.actions import AsyncLoggingAction, AsyncSetDataAction, LoggingAction, NoOpAction, SetDataAction
from hsm.core.events import Event
from hsm.core.guards import AsyncNoOpGuard, ConditionGuard, NoOpGuard
from hsm.core.state_machine import StateMachine
from hsm.core.states import State
from hsm.core.transitions import Transition
from hsm.runtime.async_support import AsyncState, AsyncStateMachine, AsyncTransition
from hsm.runtime.event_queue import AsyncEventQueue, EventQueue
from hsm.runtime.executor import Executor
from hsm.runtime.timers import AsyncTimer, Timer


# Test States
class InitialState(State):
    def on_enter(self) -> None:
        self.data["status"] = "initialized"

    def on_exit(self) -> None:
        self.data["status"] = "exiting_initial"


class ProcessingState(State):
    def on_enter(self) -> None:
        self.data["status"] = "processing"

    def on_exit(self) -> None:
        self.data["status"] = "exiting_processing"


class FinalState(State):
    def on_enter(self) -> None:
        self.data["status"] = "completed"

    def on_exit(self) -> None:
        self.data["status"] = "exiting_final"


class AsyncInitialState(AsyncState):
    async def on_enter(self) -> None:
        self.data["status"] = "initialized"

    async def on_exit(self) -> None:
        self.data["status"] = "exiting_initial"


class AsyncProcessingState(AsyncState):
    async def on_enter(self) -> None:
        self.data["status"] = "processing"

    async def on_exit(self) -> None:
        self.data["status"] = "exiting_processing"


class AsyncFinalState(AsyncState):
    async def on_enter(self) -> None:
        self.data["status"] = "completed"

    async def on_exit(self) -> None:
        self.data["status"] = "exiting_final"


# Fixtures
@pytest.fixture
def basic_states() -> List[State]:
    return [InitialState("initial"), ProcessingState("processing"), FinalState("final")]


@pytest.fixture
def basic_transitions(basic_states: List[State]) -> List[Transition]:
    return [
        Transition(
            source_id="initial",
            target_id="processing",
            guard=ConditionGuard(lambda data: data.get("status") == "initialized"),
            actions=[SetDataAction("transition_count", 1)],
        ),
        Transition(source_id="processing", target_id="final", guard=NoOpGuard(), actions=[LoggingAction("hsm.test")]),
    ]


@pytest.fixture
def state_machine(basic_states: List[State], basic_transitions: List[Transition]) -> StateMachine:
    initial_state = next(s for s in basic_states if s.get_id() == "initial")
    return StateMachine(basic_states, basic_transitions, initial_state)


@pytest.fixture
def executor(basic_states: List[State], basic_transitions: List[Transition]) -> Executor:
    initial_state = next(s for s in basic_states if s.get_id() == "initial")
    return Executor(basic_states, basic_transitions, initial_state)


@pytest.fixture
def async_states() -> List[AsyncState]:
    return [AsyncInitialState("initial"), AsyncProcessingState("processing"), AsyncFinalState("final")]


@pytest.fixture
def async_transitions(async_states: List[AsyncState]) -> List[Transition]:
    return [
        Transition(
            source_id="initial",
            target_id="processing",
            guard=AsyncNoOpGuard(),  # Using async guard for async test
            actions=[AsyncSetDataAction("transition_count", 1)],
        ),
        Transition(
            source_id="processing", target_id="final", guard=AsyncNoOpGuard(), actions=[AsyncLoggingAction("hsm.test")]
        ),
    ]


@pytest.fixture
async def async_state_machine(async_states: List[AsyncState], async_transitions: List[Transition]) -> AsyncStateMachine:
    initial_state = next(s for s in async_states if s.get_id() == "initial")
    return AsyncStateMachine(async_states, async_transitions, initial_state)


@pytest.mark.integration
class TestHSMIntegration:
    """Integration tests for the HSM library."""

    @pytest.mark.workflow
    def test_basic_state_machine_workflow(self, state_machine: StateMachine):
        """Test complete workflow through state machine."""
        # Start machine
        state_machine.start()
        assert state_machine.get_current_state_id() == "initial"

        # Process first transition
        state_machine.process_event(Event("START"))
        assert state_machine.get_current_state_id() == "processing"

        # Process second transition
        state_machine.process_event(Event("FINISH"))
        assert state_machine.get_current_state_id() == "final"

        # Verify final state
        state_machine.stop()

    @pytest.mark.workflow
    def test_executor_workflow(self, executor: Executor):
        """Test complete workflow through executor."""
        # Start executor
        executor.start()
        assert executor.get_current_state().get_id() == "initial"

        # Process events
        executor.process_event(Event("START"))
        executor.process_event(Event("FINISH"))

        # Allow time for event processing
        import time

        time.sleep(0.1)

        # Verify final state
        assert executor.get_current_state().get_id() == "final"
        executor.stop()

    @pytest.mark.asyncio
    @pytest.mark.workflow
    async def test_async_workflow(self, async_state_machine: AsyncStateMachine):
        """Test async workflow."""
        # Start machine
        await async_state_machine.start()
        assert async_state_machine.get_current_state().get_id() == "initial"

        # Process events
        await async_state_machine.process_event(Event("START"))
        await asyncio.sleep(0.1)  # Allow event processing
        assert async_state_machine.get_current_state().get_id() == "processing"

        await async_state_machine.process_event(Event("FINISH"))
        await asyncio.sleep(0.1)  # Allow event processing
        assert async_state_machine.get_current_state().get_id() == "final"

        await async_state_machine.stop()

    @pytest.mark.error_handling
    def test_error_propagation(self, state_machine: StateMachine):
        """Test error handling and propagation."""

        def failing_guard(data: Any) -> bool:
            raise ValueError("Guard failure")

        # Create transition with failing guard
        failing_transition = Transition(
            source_id="initial", target_id="processing", guard=ConditionGuard(failing_guard)
        )

        # Replace transitions
        state_machine._transitions = [failing_transition]
        state_machine.start()

        # Verify error propagation
        with pytest.raises(Exception):
            state_machine.process_event(Event("START"))

        state_machine.stop()

    @pytest.mark.recovery
    def test_state_recovery(self, state_machine: StateMachine):
        """Test recovery from failed transitions."""
        state_machine.start()

        # Set up recovery verification
        initial_data = {"key": "value"}
        state_machine._current_state.data.update(initial_data)

        def failing_action(event: Event, data: Any) -> None:
            raise RuntimeError("Action failed")

        # Create a transition that will fail
        failing_transition = Transition(source_id="initial", target_id="processing", actions=[failing_action])
        state_machine._transitions = [failing_transition]

        # Force a failed transition
        with pytest.raises(Exception):
            state_machine.process_event(Event("START"))

        # Verify state recovery
        assert state_machine.get_current_state_id() == "initial"
        assert state_machine._current_state.data["key"] == "value"

        state_machine.stop()

    @pytest.mark.resource
    def test_resource_cleanup(self, executor: Executor):
        """Test proper resource cleanup."""
        executor.start()

        # Create and use resources
        timer = Timer("test_timer", lambda t, e: None)
        event_queue = EventQueue()

        # Process some events
        executor.process_event(Event("START"))

        # Verify cleanup
        executor.stop()
        assert executor.get_current_state() is None
        assert not executor.is_running()

    @pytest.mark.performance
    def test_performance_bounds(self, executor: Executor):
        """Test performance characteristics."""
        import time

        executor.start()
        start_time = time.time()

        # Process multiple events
        for _ in range(100):
            executor.process_event(Event("START"))

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance bounds
        assert processing_time < 1.0  # Should process 100 events in under 1 second

        executor.stop()
