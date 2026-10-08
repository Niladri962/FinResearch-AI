from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, Response, status
from fastapi.responses import StreamingResponse

from app.api.deps import ContainerDep, ViewerDep
from app.models.schemas import ChatRequest, ChatResponse, ConversationDetail, ConversationOut

router = APIRouter(prefix="/chat", tags=["chat"])


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, container: ContainerDep, principal: ViewerDep):  # noqa: ANN201
    """Ask a question.

    With ``stream=true`` (default) the response is Server-Sent Events:
    ``conversation`` → ``status``/``meta`` → ``artifacts`` → ``sources`` → ``token``… → ``final``.
    The ``final`` event carries the verified answer and is authoritative.
    With ``stream=false`` a single JSON ``ChatResponse`` is returned.
    """
    if not request.stream:
        return await container.chat.respond(request, principal.subject)

    async def events() -> AsyncIterator[str]:
        async for event, data in container.chat.stream(request, principal.subject):
            yield _sse(event, data)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(container: ContainerDep, principal: ViewerDep) -> list[ConversationOut]:
    return container.chat.list_conversations(principal.subject)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, container: ContainerDep, principal: ViewerDep) -> ConversationDetail:
    return container.chat.get_conversation(conversation_id, principal.subject)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, container: ContainerDep, principal: ViewerDep) -> Response:
    container.chat.delete_conversation(conversation_id, principal.subject)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
