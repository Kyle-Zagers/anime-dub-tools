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
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_CREDIT_LINE = re.compile(
    r"^(?:edited|subtitles?|translated|translation|typeset|timing|timed|encoded|fansub|credits?)\b",
    re.IGNORECASE,
)
_CHUNK = 20
_SNIPPET = 240


class UnreadableScript(DubError):
    """The reply could not be read as one character id per speech cue.

    A diarizer label is a different error. It is not retried into a voice.
    """


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
    character = character_id(value)
    if character is None:
        raise DubError("The script returned a line with no character id.")
    return character


def character_id(value: object) -> str | None:
    """A character id, or None when the model left the field empty.

    A SPEAKER_n value is refused here. Callers do not substitute a diarizer label.
    """
    if not isinstance(value, str):
        return None
    character = value.strip()
    if not character:
        return None
    if _DIARIZER_LABEL.fullmatch(character):
        raise DubError(
            f"The script used {character}, which is a diarizer label, not a character. "
            "The model has to infer who is speaking from the dialogue. "
            "This does not fall back to unlabeled SPEAKER_n."
        )
    return character


def is_nonspeech_credit(text: str) -> bool:
    """A credit line or a cue that is only a URL.

    Dialogue that happens to mention a URL is kept. This is only for lines
    that are not speech, such as an editor credit with a link.
    """
    cleaned = light_cleanup(text)
    if not cleaned or not _URL.search(cleaned):
        return False
    if not words(_URL.sub(" ", cleaned)):
        return True
    return _CREDIT_LINE.search(cleaned) is not None


def parse_script_response(payload: str, cues: list[Cue]) -> list[ScriptLine]:
    rows = _script_rows(payload)
    by_cue = {cue.index: cue for cue in cues}
    found: dict[int, ScriptLine] = {}
    skipped: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise UnreadableScript("The script model returned a line that is not an object.")
        try:
            index = int(row["index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise UnreadableScript("The script model returned a line with no cue index.") from exc
        if index not in by_cue:
            raise UnreadableScript("The script model returned a cue index that is not in this chunk.")
        if index in found or index in skipped:
            raise UnreadableScript(f"The script model returned cue {index} more than once.")
        text = row.get("text", "")
        if not isinstance(text, str):
            text = ""
        character = character_id(row.get("character"))
        if character is None:
            if is_nonspeech_credit(by_cue[index].text):
                _skip_credit(index)
                skipped.add(index)
                continue
            raise UnreadableScript("The script returned a line with no character id.")
        found[index] = ScriptLine(index=index, character=character, text=text)
    for cue in cues:
        if cue.index in found or cue.index in skipped:
            continue
        if is_nonspeech_credit(cue.text):
            _skip_credit(cue.index)
            skipped.add(cue.index)
            continue
        raise UnreadableScript(
            "The script model did not return exactly one character id for every subtitle cue. "
            "No diarizer label was used in its place."
        )
    return [found[cue.index] for cue in cues if cue.index in found]


class ScriptLine:
    def __init__(self, index: int, character: str, text: str) -> None:
        self.index = index
        self.character = require_character(character)
        self.text = text


def apply_script(cues: list[Cue], lines: list[ScriptLine], turns: list[Turn]) -> list[AssignedCue]:
    """The script id is the voice. The cue timing stays the subtitle cue.

    A non-speech credit or URL-only cue may be absent. Every other cue needs
    the character id the model chose.
    """
    by_index = {line.index: line for line in lines}
    if len(by_index) != len(lines):
        raise DubError(
            "The script repeated a cue index. No diarizer label was used in its place."
        )
    assigned: list[AssignedCue] = []
    for cue in cues:
        line = by_index.pop(cue.index, None)
        if line is None:
            if is_nonspeech_credit(cue.text):
                continue
            raise DubError(
                "The script does not cover every subtitle cue once. "
                "No diarizer label was used in its place."
            )
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
    if by_index:
        raise DubError(
            "The script does not cover every subtitle cue once. "
            "No diarizer label was used in its place."
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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_payload(chunk, roster)},
        ]
        payload = complete(messages)
        try:
            parsed = parse_script_response(payload, chunk)
        except UnreadableScript:
            payload = complete(messages)
            try:
                parsed = parse_script_response(payload, chunk)
            except UnreadableScript as exc:
                raise DubError(_unreadable_chunk(chunk, payload, exc)) from exc
        written.extend(parsed)
        parsed_by_index = {line.index: line for line in parsed}
        for cue in chunk:
            line = parsed_by_index.get(cue.index)
            if line is None or line.character in known:
                continue
            known.add(line.character)
            roster.append({"character": line.character, "example": cue.text[:160]})
    return written


def _script_rows(payload: str) -> list:
    text = _FENCE.sub("", payload.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        salvaged = _salvage_line_objects(text)
        if not salvaged:
            raise UnreadableScript(
                "The script model did not return JSON. No speakers were invented from diarizer labels."
            )
        return salvaged
    if isinstance(data, list):
        return data
    rows = data.get("lines") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise UnreadableScript("The script model JSON has no lines array.")
    return rows


def _salvage_line_objects(payload: str) -> list[dict]:
    """Read line objects after a reply closes the lines array and keeps writing.

    json.loads rejects that document. Each line object is still valid JSON,
    so scan for those objects and keep them in order.
    """
    decoder = json.JSONDecoder()
    found: list[dict] = []
    index = 0
    length = len(payload)
    while index < length:
        start = payload.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(payload, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        if isinstance(value, dict) and "index" in value and "lines" not in value:
            found.append(value)
        index = max(end, start + 1)
    return found


def _skip_credit(index: int) -> None:
    print(f"[script] skip cue {index} non-speech credit", flush=True)


def _unreadable_chunk(chunk: list[Cue], payload: str, exc: UnreadableScript) -> str:
    first = chunk[0].index
    last = chunk[-1].index
    return (
        f"The script model reply for cues {first}-{last} is still unreadable. "
        "This does not fall back to espeak-ng, and it does not use unlabeled SPEAKER_n. "
        f"{exc} Reply: {_snippet(payload)}"
    )


def _snippet(payload: str) -> str:
    compact = re.sub(r"\s+", " ", payload).strip()
    if len(compact) <= _SNIPPET:
        return compact
    return compact[:_SNIPPET] + "..."


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
