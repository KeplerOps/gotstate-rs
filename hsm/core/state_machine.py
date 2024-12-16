# hsm/core/state_machine.py
# Copyright (c) 2024 Brad Edwards
# Licensed under the MIT License - see LICENSE file for details
# from typing import Any, Dict, List, Optional

from hsm.interfaces.abc import AbstractStateMachine
from hsm.interfaces.protocols import Event
from hsm.interfaces.types import EventID, StateID


class StateMachine(AbstractStateMachine):
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def process_event(self, event: "Event") -> None:
        raise NotImplementedError

    def get_current_state_id(self) -> StateID:
        raise NotImplementedError
