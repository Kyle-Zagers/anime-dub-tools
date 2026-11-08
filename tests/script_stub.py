"""Tiny OpenAI-compatible script server for tests. No model and no episode media."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXTURE_CHARACTERS = {
    "Look out for the signal.": "c1",
    "I heard you the first time.": "c2",
    "Then we move together.": "c1",
}


class ScriptStub:
    def __init__(self, status: int = 200, body: str | None = None, echo_authorization: bool = False) -> None:
        self.status = status
        self.body = body
        self.echo_authorization = echo_authorization
        handler = self._handler()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def endpoint(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)
        self._server.server_close()

    def _handler(self):
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                if stub.status != 200:
                    payload = stub.body or ""
                    if stub.echo_authorization:
                        payload = self.headers.get("Authorization", "")
                    encoded = payload.encode("utf-8")
                    self.send_response(stub.status)
                    self.send_header("Content-Type", "text/plain")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)
                    return
                content = _script_json(raw)
                body = json.dumps({"choices": [{"message": {"content": content}}]}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, fmt: str, *args) -> None:
                return

        return Handler


def _script_json(raw: bytes) -> str:
    request = json.loads(raw.decode("utf-8"))
    user = request["messages"][-1]["content"]
    payload = json.loads(user)
    lines = []
    for cue in payload["cues"]:
        actor = cue.get("actor")
        if isinstance(actor, str) and actor.strip():
            character = actor.strip()
        else:
            character = FIXTURE_CHARACTERS.get(cue["text"], "c1")
        lines.append({"index": cue["index"], "character": character, "text": cue["text"]})
    return json.dumps({"lines": lines})
