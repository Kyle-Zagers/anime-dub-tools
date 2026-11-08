"""Turn English subtitle cues into a script with one stable id per character.

The language model infers who is speaking. Diarizer labels are not identities.
The spoken words stay the subtitle line, aside from markup that cannot be said.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from anime_dub_tools.assign import fit_window
from anime_dub_tools.models import AssignedCue, Cue, DubError, Turn

_DIARIZER_LABEL = re.compile(r"SPEAKER_\d+\Z", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9']+")
_LEADING_DASH = re.compile(r"^(?:[-\u2013\u2014]+\s*)+")
_NOTES = re.compile(r"[♪♫]+")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_CHUNK = 20

SYSTEM_PROMPT = """You write a dubbing script from English subtitle cues.

Read the cues in order. For each cue, decide who is speaking.
Infer that from the dialogue: names people say, how they address each other, and who answers whom.
The same person keeps one character id for the whole episode, including later lines that say no name.
Ids are short labels you choose, such as c1 or c2. When the roster already has that person, reuse the id.
Do not use labels of the form SPEAKER_n. Those are not characters.
A name in an actor field is only a hint from the subtitle file. The dialogue decides.
Do not rewrite the line. Copy its words. Change text only when it cannot be spoken as written, such as leftover markup.
Do not paraphrase, summarize, or change the meaning.
Return JSON only, with one object per cue and the same indexes:
{"lines": [{"index": 1, "character": "c1", "text": "the line"}]}
"""


def script_configuration_error() -> str:
    return (
        "No script model is configured, so the episode cannot be dubbed. "
        "A language model has to read the English cues and give each line a stable character id. "
        "Point --script-model or ANIME_DUB_SCRIPT_MODEL at a GGUF file on this machine. "
        "An OpenAI-compatible server is optional and stays off until you set "
        "--script-endpoint or ANIME_DUB_SCRIPT_ENDPOINT, with --script-model as the name that server expects. "
        "This does not fall back to espeak-ng, and it does not use unlabeled SPEAKER_n as a script."
    )


def light_cleanup(text: str) -> str:
    """Drop markup that cannot be spoken. Leave the words alone."""
    cleaned = text.replace("\n", " ").replace("\r", " ")
    cleaned = _NOTES.sub(" ", cleaned)
    cleaned = _LEADING_DASH.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD.finditer(text)]


def faithful_spoken(original: str, proposed: str | None) -> str:
    """Speak the subtitle line. A paraphrase is discarded.

    Leading dialogue dashes and music notes are not words, so those can go.
    Any other change of wording is ignored.
    """
    cleaned = light_cleanup(original)
    if proposed is None or not proposed.strip():
        return cleaned
    if words(proposed) == words(cleaned):
        return cleaned
    return cleaned


def require_character(value: object) -> str:
    if not isinstance(value, str):
        raise DubError("The script returned a line with no character id.")
    character = value.strip()
    if not character:
        raise DubError("The script returned a line with no character id.")
    if _DIARIZER_LABEL.fullmatch(character):
        raise DubError(
            f"The script used {character}, which is a diarizer label, not a character. "
            "The model has to infer who is speaking from the dialogue. "
            "This does not fall back to unlabeled SPEAKER_n."
        )
    return character


def parse_script_response(payload: str, cues: list[Cue]) -> list[ScriptLine]:
    try:
        data = json.loads(_FENCE.sub("", payload.strip()))
    except json.JSONDecodeError as exc:
        raise DubError(
            "The script model did not return JSON. No speakers were invented from diarizer labels."
        ) from exc
    rows = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise DubError("The script model JSON has no lines array.")
    found: dict[int, ScriptLine] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise DubError("The script model returned a line that is not an object.")
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DubError("The script model returned a line with no cue index.") from exc
        text = row.get("text", "")
        if not isinstance(text, str):
            text = ""
        found[index] = ScriptLine(index=index, character=require_character(row.get("character")), text=text)
    expected = [cue.index for cue in cues]
    if sorted(found) != sorted(expected):
        raise DubError(
            "The script model did not return exactly one character id for every subtitle cue. "
            "No diarizer label was used in its place."
        )
    return [found[index] for index in expected]


class ScriptLine:
    def __init__(self, index: int, character: str, text: str) -> None:
        self.index = index
        self.character = require_character(character)
        self.text = text


def apply_script(cues: list[Cue], lines: list[ScriptLine], turns: list[Turn]) -> list[AssignedCue]:
    """The script id is the voice. The cue timing stays the subtitle cue."""
    by_index = {line.index: line for line in lines}
    if len(by_index) != len(lines) or sorted(by_index) != sorted(cue.index for cue in cues):
        raise DubError(
            "The script does not cover every subtitle cue once. "
            "No diarizer label was used in its place."
        )
    assigned: list[AssignedCue] = []
    for cue in cues:
        line = by_index[cue.index]
        spoken = faithful_spoken(cue.text, line.text)
        if line.text.strip() and words(line.text) != words(spoken):
            print(f"[script] line {cue.index} kept the subtitle wording", flush=True)
        start, end = fit_window(cue, turns)
        assigned.append(
            AssignedCue(
                cue=cue,
                speaker=line.character,
                start=start,
                end=end,
                spoken=spoken,
            )
        )
    return assigned


Complete = Callable[[list[dict[str, str]]], str]


def script_from_complete(complete: Complete, cues: list[Cue], chunk_size: int = _CHUNK) -> list[ScriptLine]:
    """Ask the model in order, carrying the ids already chosen.

    One character keeps the same id across chunks because later requests see
    the roster. The model is still the thing that chooses the id.
    """
    if chunk_size < 1:
        raise DubError("Script chunks must contain at least one cue.")
    roster: list[dict[str, str]] = []
    written: list[ScriptLine] = []
    known = {item["character"] for item in roster}
    for start in range(0, len(cues), chunk_size):
        chunk = cues[start : start + chunk_size]
        payload = complete(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_payload(chunk, roster)},
            ]
        )
        parsed = parse_script_response(payload, chunk)
        written.extend(parsed)
        for line, cue in zip(parsed, chunk):
            if line.character in known:
                continue
            known.add(line.character)
            roster.append({"character": line.character, "example": cue.text[:160]})
    return written


def _user_payload(cues: list[Cue], roster: list[dict[str, str]]) -> str:
    packed = []
    for cue in cues:
        row: dict[str, object] = {
            "index": cue.index,
            "start": round(cue.start, 3),
            "end": round(cue.end, 3),
            "text": cue.text,
        }
        if cue.actor:
            row["actor"] = cue.actor
        packed.append(row)
    return json.dumps({"roster": roster, "cues": packed}, ensure_ascii=False)


def build_script_writer(
    model: str | None,
    endpoint: str | None,
    api_key: str | None = None,
) -> ScriptWriter:
    """Local GGUF unless the user pointed at an OpenAI-compatible server."""
    model = _blank(model)
    endpoint = _blank(endpoint)
    api_key = _blank(api_key)
    if endpoint:
        if model is None:
            raise DubError(
                "A script endpoint is set, but no model name was given. "
                "Pass --script-model or ANIME_DUB_SCRIPT_MODEL with the name that server expects."
            )
        return EndpointScriptWriter(endpoint, model, api_key)
    if model is None:
        raise DubError(script_configuration_error())
    path = Path(model)
    if path.suffix.lower() != ".gguf":
        raise DubError(
            f"Script model {model!r} is not a local GGUF file, and no OpenAI-compatible endpoint is set. "
            "Pass a .gguf path with --script-model or ANIME_DUB_SCRIPT_MODEL so the subtitle text stays on this machine. "
            "Set --script-endpoint or ANIME_DUB_SCRIPT_ENDPOINT only if you want that optional server. "
            "This tool does not download a language model."
        )
    if not path.is_file():
        raise DubError(
            f"Local script model was not found at {path}. "
            "Point --script-model or ANIME_DUB_SCRIPT_MODEL at a GGUF file on this machine, "
            "or set --script-endpoint / ANIME_DUB_SCRIPT_ENDPOINT. "
            "This tool does not download a language model."
        )
    return LlamaScriptWriter(path)


class ScriptWriter:
    def write(self, cues: list[Cue]) -> list[ScriptLine]:
        raise NotImplementedError


class EndpointScriptWriter(ScriptWriter):
    def __init__(self, endpoint: str, model: str, api_key: str | None) -> None:
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key

    def write(self, cues: list[Cue]) -> list[ScriptLine]:
        return script_from_complete(self._complete, cues)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        url = _chat_url(self.endpoint)
        body = json.dumps(
            {"model": self.model, "temperature": 0, "messages": messages},
            ensure_ascii=False,
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            detail = _redact(detail, self.api_key)
            raise DubError(f"Script endpoint returned HTTP {exc.code}. {detail}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DubError(
                f"Could not reach the script endpoint at {url}. "
                "Check --script-endpoint or ANIME_DUB_SCRIPT_ENDPOINT."
            ) from exc
        return _message_text(raw)


class LlamaScriptWriter(ScriptWriter):
    """In-process GGUF. Layers stay on CPU so the dub GPU is left to the voice model."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._llm = None

    def write(self, cues: list[Cue]) -> list[ScriptLine]:
        return script_from_complete(self._complete, cues)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        llm = self._load()
        try:
            result = llm.create_chat_completion(messages=messages, temperature=0)
        except Exception as exc:
            raise DubError(
                f"The local script model at {self.model_path} failed while reading the cues. {exc}"
            ) from exc
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise DubError("The local script model returned no script text.") from exc
        if not isinstance(content, str) or not content.strip():
            raise DubError("The local script model returned no script text.")
        return content

    def _load(self):
        if self._llm is not None:
            return self._llm
        if not self.model_path.is_file():
            raise DubError(
                f"Local script model was not found at {self.model_path}. "
                "This tool does not download a language model."
            )
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise DubError(
                "A local GGUF script model was set, but llama-cpp-python is not installed.\n"
                "  pip install llama-cpp-python\n"
                "This tool does not download the GGUF."
            ) from exc
        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=8192,
                n_gpu_layers=0,
                verbose=False,
            )
        except Exception as exc:
            raise DubError(
                f"Could not load the script model at {self.model_path}. "
                "The file is missing or unreadable. This tool does not download a language model. "
                f"{exc}"
            ) from exc
        return self._llm


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _chat_url(endpoint: str) -> str:
    url = endpoint.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url + "/v1/chat/completions"


def _message_text(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, UnicodeError) as exc:
        raise DubError("The script endpoint returned no script text.") from exc
    if not isinstance(content, str) or not content.strip():
        raise DubError("The script endpoint returned no script text.")
    return content


def _redact(detail: str, api_key: str | None) -> str:
    if api_key and api_key in detail:
        return detail.replace(api_key, "[redacted]")
    return detail
