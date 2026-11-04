"""Cancellation must abandon only the request bound to the cancelled response."""

from unittest.mock import AsyncMock, MagicMock, call

import pytest
from openai.types.realtime import (
    RealtimeResponse,
    ResponseDoneEvent,
    ResponseCreatedEvent,
    ConversationItemCreatedEvent,
    RealtimeConversationItemUserMessage,
)

from reachy_mini_pose_coach.cancellation_context import CancellationContext


def user_item(item_id: str) -> ConversationItemCreatedEvent:
    """Build a real SDK event for acknowledged conversational input."""
    return ConversationItemCreatedEvent(
        type="conversation.item.created",
        event_id="event_" + item_id,
        item=RealtimeConversationItemUserMessage(type="message", role="user", id=item_id, content=[]),
    )


def response_created(response_id: str) -> ResponseCreatedEvent:
    """Build the response event that binds pending inputs."""
    return ResponseCreatedEvent(
        type="response.created",
        event_id="created_" + response_id,
        response=RealtimeResponse(id=response_id, status="in_progress"),
    )


def response_done(response_id: str, *, cancelled: bool) -> ResponseDoneEvent:
    """Build a terminal response event."""
    return ResponseDoneEvent(
        type="response.done",
        event_id="done_" + response_id,
        response=RealtimeResponse(id=response_id, status="cancelled" if cancelled else "completed"),
    )


@pytest.mark.asyncio
async def test_cancellation_deletes_audio_and_snapshot_but_preserves_newer_and_completed_input():
    """Input ownership survives new speech arriving before the cancellation terminal."""
    context = CancellationContext()
    connection = MagicMock()
    connection.conversation.item.delete = AsyncMock()
    for event in (
        user_item("completed_input"),
        response_created("completed"),
        response_done("completed", cancelled=False),
        user_item("old_audio"),
        user_item("old_snapshot"),
        user_item("old_snapshot"),
        response_created("cancelled"),
        user_item("new_audio"),
        response_done("cancelled", cancelled=True),
        response_done("cancelled", cancelled=True),
    ):
        await context.handle_event(event, connection)
    assert connection.conversation.item.delete.await_args_list == [
        call(item_id="old_audio"),
        call(item_id="old_snapshot"),
    ]
    await context.handle_event(response_created("next"), connection)
    await context.handle_event(response_done("next", cancelled=True), connection)
    assert connection.conversation.item.delete.await_args_list[-1] == call(item_id="new_audio")


@pytest.mark.asyncio
async def test_unknown_response_cannot_delete_pending_input():
    """A late terminal from an untracked response cannot touch the current request."""
    context = CancellationContext()
    connection = MagicMock()
    connection.conversation.item.delete = AsyncMock()
    await context.handle_event(user_item("new_audio"), connection)
    await context.handle_event(response_done("unknown", cancelled=True), connection)
    connection.conversation.item.delete.assert_not_called()


@pytest.mark.asyncio
async def test_cleanup_transport_failure_propagates_to_session_recovery():
    """Do not release the response sender as though cleanup succeeded on a broken connection."""
    context = CancellationContext()
    connection = MagicMock()
    connection.conversation.item.delete = AsyncMock(side_effect=OSError("disconnected"))
    await context.handle_event(user_item("old_audio"), connection)
    await context.handle_event(response_created("cancelled"), connection)
    with pytest.raises(OSError, match="disconnected"):
        await context.handle_event(response_done("cancelled", cancelled=True), connection)
