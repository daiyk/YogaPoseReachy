"""Same-origin coach panel routes, scheduling state changes on the conversation loop."""

import asyncio
from pathlib import Path
from urllib.parse import urlparse
from collections.abc import Callable

from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel, ConfigDict

from .bridge import CoachBridge
from .settings import CoachSettings, save_coach_settings


class PracticeRequest(BaseModel):
    """The explicit practice state requested by the panel."""

    model_config = ConfigDict(extra="forbid", strict=True)
    active: bool


class ReferenceRequest(BaseModel):
    """A saved reference selection and anatomical comparison side."""

    model_config = ConfigDict(extra="forbid", strict=True)
    selection: str
    side: str = "same"
    parts: list[str] | None = None


def register_coach_routes(
    app: FastAPI,
    get_coach: Callable[[], CoachBridge | None],
    get_loop: Callable[[], asyncio.AbstractEventLoop | None],
    directory: Path,
) -> None:
    """Mount bounded controls and demand-only preview on the existing app server."""
    mutation_lock = asyncio.Lock()

    def check_origin(request: Request) -> None:
        origin = request.headers.get("origin")
        if request.headers.get("X-Reachy-Coach") != "1" or (
            origin and urlparse(origin).netloc != request.headers.get("host")
        ):
            raise HTTPException(403, "Use the coach panel on the same host")

    async def on_loop(operation: Callable[[CoachBridge], dict[str, object]]) -> dict[str, object]:
        async def run() -> dict[str, object]:
            coach = get_coach()
            if coach is None:
                raise HTTPException(503, "The coach is still starting")
            return operation(coach)

        loop = get_loop()
        if loop is None or not loop.is_running():
            raise HTTPException(503, "The conversation loop is not running")
        if loop is asyncio.get_running_loop():
            return await run()
        try:
            return await asyncio.wait_for(
                asyncio.wrap_future(asyncio.run_coroutine_threadsafe(run(), loop)), timeout=5
            )
        except TimeoutError as exc:
            raise HTTPException(503, "The coach is busy; try again") from exc

    @app.get("/api/coach/status")
    async def status(request: Request) -> dict[str, object]:
        check_origin(request)
        return await on_loop(lambda coach: coach.panel_state())

    @app.post("/api/coach/practice")
    async def practice(body: PracticeRequest, request: Request) -> dict[str, object]:
        check_origin(request)

        def change(coach: CoachBridge) -> dict[str, object]:
            coach.set_practice(body.active)
            return coach.panel_state()

        return await on_loop(change)

    @app.put("/api/coach/settings")
    async def settings(body: CoachSettings, request: Request) -> dict[str, object]:
        check_origin(request)

        def change(coach: CoachBridge) -> dict[str, object]:
            coach.configure(body)
            return coach.panel_state()

        async with mutation_lock:
            await on_loop(lambda coach: {})
            await asyncio.to_thread(save_coach_settings, directory, body)
            return await on_loop(change)

    @app.get("/api/coach/reference")
    async def reference(request: Request) -> Response:
        check_origin(request)
        coach = get_coach()
        jpeg = coach.service.reference_image() if coach is not None and coach.service is not None else b""
        return Response(
            jpeg,
            media_type="image/jpeg",
            status_code=200 if jpeg else 204,
            headers={"Cache-Control": "no-store", "Cross-Origin-Resource-Policy": "same-origin"},
        )

    @app.post("/api/coach/reference/select")
    async def select(body: ReferenceRequest, request: Request) -> dict[str, object]:
        check_origin(request)
        return await change_reference(body.selection, body.side, parts=body.parts)

    async def change_reference(
        selection: str, side: str, content: bytes | None = None, parts: list[str] | None = None
    ) -> dict[str, object]:
        async with mutation_lock:
            return await apply_reference(selection, side, content, parts)

    async def apply_reference(
        selection: str, side: str, content: bytes | None, parts: list[str] | None
    ) -> dict[str, object]:
        await on_loop(lambda coach: {})
        coach = get_coach()
        if coach is None or coach.service is None:
            raise HTTPException(503, "Live pose service is unavailable")
        try:
            reference_id = await asyncio.to_thread(coach.service.select_reference, selection, side, content, parts)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, "Could not save the reference selection") from exc
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

        def change(current: CoachBridge) -> dict[str, object]:
            current.set_reference(reference_id)
            return current.panel_state()

        return await on_loop(change)

    @app.post("/api/coach/reference/upload")
    async def upload(request: Request) -> dict[str, object]:
        check_origin(request)
        content = bytearray()
        async for chunk in request.stream():
            if len(content) + len(chunk) > 5 * 1024 * 1024:
                raise HTTPException(413, "Choose a JPEG or PNG of at most 5 MB")
            content.extend(chunk)
        return await change_reference("uploaded", "same", bytes(content))

    @app.get("/api/coach/preview", response_model=None)
    async def preview(request: Request) -> dict[str, object] | Response:
        check_origin(request)
        coach = get_coach()
        if coach is None or coach.service is None:
            return Response(status_code=204)
        packet = await asyncio.to_thread(coach.service.preview)
        return packet if packet is not None else Response(status_code=204)
