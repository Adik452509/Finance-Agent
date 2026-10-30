"""The frontend's only door to the backend: plain HTTP, plus Server-Sent Events for live answers.

Every request carries the shared X-Api-Token and the signed-in user's identity; the backend
trusts the identity only because of the token (see src/api/app.py).
"""

import json
from collections.abc import Iterator
from urllib.parse import quote

import httpx


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


class BackendDown(RuntimeError):
    """The backend is not running or not reachable."""


def parse_sse(lines: Iterator[str]) -> Iterator[dict]:
    """Server-Sent Events -> event dicts (one 'data:' line per event, as the backend sends)."""
    for line in lines:
        if line.startswith("data: "):
            try:
                yield json.loads(line[len("data: "):])
            except json.JSONDecodeError:
                continue


class ApiClient:
    def __init__(self, base_url: str, email: str, name: str | None = None, token: str | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        headers = {"X-User-Email": email}
        if name:
            headers["X-User-Name"] = quote(name)  # HTTP headers are ASCII: names may not be
        if token:
            headers["X-Api-Token"] = token
        self._http = httpx.Client(base_url=base_url, headers=headers, timeout=30, transport=transport)

    def _call(self, method: str, path: str, **kwargs):
        try:
            response = self._http.request(method, path, **kwargs)
        except httpx.TransportError as e:
            raise BackendDown(f"The research service is not reachable ({type(e).__name__}).") from e
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(response.status_code, detail if isinstance(detail, str) else json.dumps(detail))
        return response.json() if response.content else None

    # -- account and chats --
    def me(self) -> dict:
        return self._call("GET", "/me")

    def status(self) -> dict:
        return self._call("GET", "/status")

    def conversations(self) -> list[dict]:
        return self._call("GET", "/conversations")

    def conversation(self, conversation_id: str) -> dict:
        return self._call("GET", f"/conversations/{conversation_id}")

    def new_conversation(self) -> dict:
        return self._call("POST", "/conversations", json={})

    def rename(self, conversation_id: str, title: str) -> dict:
        return self._call("PATCH", f"/conversations/{conversation_id}", json={"title": title})

    def delete(self, conversation_id: str) -> None:
        self._call("DELETE", f"/conversations/{conversation_id}")

    def feedback(self, message_id: str, rating: int, comment: str | None = None) -> None:
        self._call("POST", f"/messages/{message_id}/feedback", json={"rating": rating, "comment": comment})

    def ask(self, conversation_id: str, question: str) -> Iterator[dict]:
        """Stream the research as events; the last one is "answer" (or an error)."""
        try:
            with self._http.stream("POST", f"/conversations/{conversation_id}/messages",
                                   json={"question": question}, timeout=httpx.Timeout(30, read=400)) as response:
                if response.status_code >= 400:
                    response.read()
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    raise ApiError(response.status_code, str(detail))
                yield from parse_sse(response.iter_lines())
        except httpx.TransportError as e:
            raise BackendDown(f"The connection to the research service was lost ({type(e).__name__}).") from e

    # -- library and admin --
    def companies(self) -> list[dict]:
        return self._call("GET", "/companies")

    def add_company(self, body: dict) -> dict:
        return self._call("POST", "/admin/companies", json=body)

    def jobs(self) -> list[dict]:
        return self._call("GET", "/admin/jobs")

    def reload(self) -> dict:
        return self._call("POST", "/admin/reload")
