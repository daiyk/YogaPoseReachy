"""Discard unanswered inputs when a Hugging Face response is cancelled."""

import logging

from openai.types.realtime import RealtimeServerEvent
from openai.resources.realtime.realtime import AsyncRealtimeConnection


logger = logging.getLogger(__name__)


class CancellationContext:
    """Bind acknowledged user items to responses without touching newer input."""

    def __init__(self) -> None:
        """Start with an empty connection-local history ledger."""
        self._pending_inputs: dict[str, None] = {}
        self._response_inputs: dict[str, tuple[str, ...]] = {}
        self._requested_inputs: tuple[str, ...] | None = None

    def prepare_response(self) -> None:
        """Freeze inputs at the client send, before a late response.created can include newer speech."""
        self._requested_inputs = tuple(self._pending_inputs)

    def response_rejected(self) -> None:
        """Discard the binding for an explicitly rejected response request."""
        self._requested_inputs = None

    async def handle_event(self, event: RealtimeServerEvent, connection: AsyncRealtimeConnection) -> None:
        """Remove a cancelled request before the next queued response can start.

        HF rolls back provisional assistant output on cancellation. Leaving its
        unanswered user items behind can replay that request on the next turn.
        Completed history stays intact; an interrupted request is abandoned.
        """
        if event.type in ("conversation.item.created", "conversation.item.added"):
            item = event.item
            if item.type == "message" and item.role == "user" and item.id:
                self._pending_inputs[item.id] = None
        elif event.type == "response.created" and event.response.id:
            self._response_inputs[event.response.id] = (
                self._requested_inputs if self._requested_inputs is not None else tuple(self._pending_inputs)
            )
            self._requested_inputs = None
        elif event.type == "response.done" and event.response.id:
            item_ids = self._response_inputs.pop(event.response.id, ())
            if event.response.status == "cancelled":
                for item_id in item_ids:
                    await connection.conversation.item.delete(item_id=item_id)
                logger.info("Discarded %d input items from cancelled response %s", len(item_ids), event.response.id)
            for item_id in item_ids:
                self._pending_inputs.pop(item_id, None)
