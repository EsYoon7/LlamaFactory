"""AgentNet (OpenCUA) SFT dataset converter, aligned with OSWorld PR #448
(``mm_agents/qwen35vl_agent.py``).

Converts raw AgentNet trajectories (``agentnet_ubuntu_5k.jsonl`` /
``agentnet_win_mac_18k.jsonl``) into LLaMA-Factory sharegpt samples whose
rendered chat string exactly matches what the eval agent produces at
inference time, so SFT targets live on the same manifold as the test-time
prompts.

Alignment points (all derived from the PR #448 agent source):
  * System message: baked directly into each sample (persona + wrapped
    ``computer_use`` tool JSON + ``<IMPORTANT>`` reminder block +
    ``# Response format`` block). ``tools`` field in the parquet is left
    empty so LLaMA-Factory's ``qwen3_5`` template does not append a *second*
    tool prompt on top.
  * First user turn: ``<image>\\n<instruction_prompt>`` where
    ``instruction_prompt`` uses the PR #448 wording
    (``"Please generate the next move…\\n\\nInstruction: {…}\\n\\n
    Previous actions:\\n"``). ``Previous actions`` is intentionally empty —
    at inference the agent never populates this list either; prior actions
    flow through the conversation history.
  * Subsequent screenshot turns: ``role="user"`` with content wrapped in
    ``<tool_response>\\n{image_or_text}\\n</tool_response>`` (the agent does
    this wrapping *inside* the user content, so LLaMA-Factory renders it
    with the plain ``format_user`` slot).
  * Collapse placeholder for screenshots outside the image-history window:
    ``"This screenshot has been collapsed."`` (verbatim from the agent).
  * Assistant turn: ``<think>{thought}</think>\\n\\nAction: {action_nl}\\n
    <tool_call>{"name":"computer_use","arguments":{…}}</tool_call>``. The
    JSON payload is re-emitted as the XML-nested
    ``<function=…>/<parameter=…>`` form by ``Qwen35ToolUtils`` at tokenize
    time, matching the ``--tool-call-parser qwen3_coder`` parser the agent
    expects.

AgentNet-specific notes:
  * AgentNet encodes a single action per step inside a PyAutoGUI /
    ``computer.*`` *code string*; we AST-parse it into the ComputerUse
    action dict here.
  * Coordinates in the source code are already **relative [0..1]**, so the
    rescale to 1000×1000 is just ``int(round(x * 1000))``.
  * AgentNet only uses the 14 base actions (no ``visit_url`` /
    ``web_search`` / etc.), so we emit the base tool schema here rather
    than ``extend_schema``.

Public entry point: :func:`convert_agentnet_specs`.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any

if __package__:
    from .web_agent_cu import (  # type: ignore[no-redef]
        DISPLAY_H,
        DISPLAY_W,
        IMAGE_OMITTED_TEXT,
        IMAGE_PLACEHOLDER,
        _escape_image_tag,
        _format_reasoning,
        build_tool_schema,
    )
else:  # standalone load (e.g. scripts/agentnet/preprocess.py)
    import importlib.util as _u
    from pathlib import Path as _P
    _m = _u.module_from_spec(_u.spec_from_file_location(
        "_agentnet_webcu", _P(__file__).with_name("web_agent_cu.py")
    ))
    _u.spec_from_file_location("_agentnet_webcu", _P(__file__).with_name("web_agent_cu.py")).loader.exec_module(_m)
    DISPLAY_H = _m.DISPLAY_H
    DISPLAY_W = _m.DISPLAY_W
    IMAGE_OMITTED_TEXT = _m.IMAGE_OMITTED_TEXT
    IMAGE_PLACEHOLDER = _m.IMAGE_PLACEHOLDER
    _escape_image_tag = _m._escape_image_tag
    _format_reasoning = _m._format_reasoning
    build_tool_schema = _m.build_tool_schema


# -----------------------------------------------------------------------------
# AgentNet code-string parser
# -----------------------------------------------------------------------------

def _to_abs_xy(x: float, y: float) -> list[int]:
    """Scale relative [0,1] → [0, DISPLAY_*]. Clamps to the display box."""
    nx = max(0, min(DISPLAY_W, int(round(float(x) * DISPLAY_W))))
    ny = max(0, min(DISPLAY_H, int(round(float(y) * DISPLAY_H))))
    return [nx, ny]


def _const(node: ast.AST) -> Any:
    """Evaluate a constant-only AST node (Constant / List / Tuple / Dict)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_const(e) for e in node.elts]
    if isinstance(node, ast.Dict):
        return {_const(k): _const(v) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_const(node.operand)
    raise ValueError(f"non-constant AST node: {ast.dump(node)}")


def _kwargs(call: ast.Call) -> dict[str, Any]:
    return {kw.arg: _const(kw.value) for kw in call.keywords if kw.arg is not None}


def _pos(call: ast.Call) -> list[Any]:
    return [_const(a) for a in call.args]


def _call_fn_name(call: ast.Call) -> str:
    """Return a dotted name like ``pyautogui.click`` or ``computer.terminate``."""
    node = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _xy_from(call: ast.Call) -> tuple[float, float] | None:
    kw = _kwargs(call)
    if "x" in kw and "y" in kw:
        return float(kw["x"]), float(kw["y"])
    pos = _pos(call)
    if len(pos) >= 2:
        return float(pos[0]), float(pos[1])
    return None


def parse_agentnet_code(code: str) -> list[dict[str, Any]]:
    """Parse an AgentNet ``code`` string into zero or more computer_use calls.

    Returns a list of ``{"name": "computer_use", "arguments": {...}}``.
    Returns ``[]`` for no-ops or unparsable code (caller decides what to do).
    """
    code = (code or "").strip()
    if not code:
        return []
    try:
        mod = ast.parse(code, mode="exec")
    except SyntaxError:
        return []

    def call(args: dict[str, Any]) -> dict[str, Any]:
        return {"name": "computer_use", "arguments": args}

    out: list[dict[str, Any]] = []
    for stmt in mod.body:
        if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
            continue
        c = stmt.value
        name = _call_fn_name(c)
        short = name.split(".")[-1]
        try:
            if short in ("click", "leftClick"):
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "left_click", "coordinate": _to_abs_xy(*xy)}))
            elif short == "doubleClick":
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "double_click", "coordinate": _to_abs_xy(*xy)}))
            elif short == "tripleClick":
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "triple_click", "coordinate": _to_abs_xy(*xy)}))
            elif short == "rightClick":
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "right_click", "coordinate": _to_abs_xy(*xy)}))
            elif short == "middleClick":
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "middle_click", "coordinate": _to_abs_xy(*xy)}))
            elif short == "moveTo":
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "mouse_move", "coordinate": _to_abs_xy(*xy)}))
            elif short == "dragTo":
                # pyautogui.dragTo(x, y, button='left') → single left_click_drag
                xy = _xy_from(c)
                if xy is not None:
                    out.append(call({"action": "left_click_drag", "coordinate": _to_abs_xy(*xy)}))
            elif short == "write":
                kw = _kwargs(c)
                text = kw.get("message")
                if text is None:
                    pos = _pos(c)
                    text = pos[0] if pos else ""
                out.append(call({"action": "type", "text": str(text)}))
            elif short == "typewrite":
                pos = _pos(c)
                text = pos[0] if pos else _kwargs(c).get("message", "")
                out.append(call({"action": "type", "text": str(text)}))
            elif short == "press":
                pos = _pos(c)
                keys = pos[0] if pos else _kwargs(c).get("keys", "")
                if isinstance(keys, list):
                    out.append(call({"action": "key", "keys": [str(k) for k in keys]}))
                else:
                    out.append(call({"action": "key", "keys": [str(keys)]}))
            elif short == "hotkey":
                pos = _pos(c)
                if len(pos) == 1 and isinstance(pos[0], list):
                    keys = pos[0]
                else:
                    keys = pos
                out.append(call({"action": "key", "keys": [str(k) for k in keys]}))
            elif short == "scroll":
                pos = _pos(c)
                amount = int(pos[0]) if pos else 0
                out.append(call({"action": "scroll", "pixels": amount}))
            elif short == "hscroll":
                pos = _pos(c)
                amount = int(pos[0]) if pos else 0
                out.append(call({"action": "hscroll", "pixels": amount}))
            elif short == "wait":
                kw = _kwargs(c)
                pos = _pos(c)
                t = kw.get("seconds") or kw.get("time") or (pos[0] if pos else 1)
                out.append(call({"action": "wait", "time": float(t)}))
            elif short == "terminate":
                kw = _kwargs(c)
                status = kw.get("status", "success")
                out.append(call({"action": "terminate", "status": str(status)}))
            elif short == "answer":
                kw = _kwargs(c)
                pos = _pos(c)
                text = kw.get("text") or (pos[0] if pos else "")
                out.append(call({"action": "answer", "text": str(text)}))
            # else: unrecognised call (e.g. pyautogui.keyDown); silently skip
        except Exception:
            continue
    return out


# -----------------------------------------------------------------------------
# Trajectory spec + sample builder
# -----------------------------------------------------------------------------

@dataclass
class AgentNetStep:
    thought: str
    action_nl: str     # natural-language action description; folded into reasoning
    code: str
    image_path: str


@dataclass
class AgentNetSpec:
    task_id: str
    instruction: str
    steps: list[AgentNetStep]


# -----------------------------------------------------------------------------
# OSWorld PR #448 alignment constants (verbatim from mm_agents/qwen35vl_agent.py)
# -----------------------------------------------------------------------------

OSWORLD_COLLAPSE_TEXT = "This screenshot has been collapsed."

OSWORLD_INSTRUCTION_PROMPT_TMPL = (
    "Please generate the next move according to the UI screenshot, "
    "instruction and previous actions.\n\n"
    "Instruction: {instruction}\n\n"
    "Previous actions:\n"
)

_OSWORLD_PERSONA = (
    "You are a multi-purpose intelligent assistant. "
    "Based on my requests, you can use tools to help me complete various tasks."
)

_OSWORLD_DESCRIPTION_PROMPT = (
    "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
    "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
    "You must click on desktop icons to start applications.\n"
    "* Some applications may take time to start or process actions, so you may need to wait and take "
    "successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window "
    "doesn't open, try wait and taking another screenshot.\n"
    "* The screen's resolution is 1000x1000.\n"
    "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
    "screenshot to determine the coordinates of the element before moving the cursor.\n"
    "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting "
    "your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
    "Don't click boxes on their edges."
)

_OSWORLD_ACTION_DESCRIPTION = (
    "The action to perform. The available actions are:\n"
    "* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\n"
    "* `type`: Type a string of text on the keyboard.\n"
    "* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n"
    "* `left_click`: Click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
    "* `left_click_drag`: Click and drag the cursor to a specified (x, y) pixel coordinate on the screen.\n"
    "* `right_click`: Click the right mouse button at a specified (x, y) pixel coordinate on the screen.\n"
    "* `middle_click`: Click the middle mouse button at a specified (x, y) pixel coordinate on the screen.\n"
    "* `double_click`: Double-click the left mouse button at a specified (x, y) pixel coordinate on the screen.\n"
    "* `triple_click`: Triple-click the left mouse button at a specified (x, y) pixel coordinate on the screen "
    "(simulated as double-click since it's the closest action).\n"
    "* `scroll`: Performs a scroll of the mouse scroll wheel.\n"
    "* `hscroll`: Performs a horizontal scroll (mapped to regular scroll).\n"
    "* `wait`: Wait specified seconds for the change to happen.\n"
    "* `terminate`: Terminate the current task and report its completion status.\n"
    "* `answer`: Answer a question."
)


def _build_osworld_tools_def() -> dict[str, Any]:
    """Tool schema verbatim from PR #448 agent (wrapped in ``{"type":"function",...}``)."""
    return {
        "type": "function",
        "function": {
            "name": "computer_use",
            "description": _OSWORLD_DESCRIPTION_PROMPT,
            "parameters": {
                "type": "object",
                "required": ["action"],
                "properties": {
                    "action": {
                        "type": "string",
                        "description": _OSWORLD_ACTION_DESCRIPTION,
                        "enum": [
                            "key", "type", "mouse_move", "left_click", "left_click_drag",
                            "right_click", "middle_click", "double_click", "triple_click",
                            "scroll", "hscroll", "wait", "terminate", "answer",
                        ],
                    },
                    "keys": {"type": "array", "description": "Required only by `action=key`."},
                    "text": {
                        "type": "string",
                        "description": (
                            "Required by `action=type` and `action=answer`. "
                            "Optional for click actions (left_click, right_click, middle_click, "
                            "double_click, triple_click) to specify modifier keys "
                            "(e.g., 'ctrl', 'shift', 'ctrl+shift'). Optional for scroll actions "
                            "(scroll, hscroll) to specify a modifier key (e.g., 'shift', 'ctrl') "
                            "to hold during scrolling."
                        ),
                    },
                    "coordinate": {"type": "array", "description": "(x, y) coordinates."},
                    "pixels": {"type": "number", "description": "Scroll amount."},
                    "time": {"type": "number", "description": "Seconds to wait."},
                    "status": {
                        "type": "string",
                        "description": "Task status for terminate.",
                        "enum": ["success", "failure"],
                    },
                },
            },
        },
    }


def _build_osworld_system_prompt(today: _dt.date | None = None) -> str:
    """Reproduce the agent's system prompt string. Date is frozen at preprocess
    time; minor drift from the eval-day date is acceptable (system prompts
    of this size are robust to a date substitution)."""
    tools_json = json.dumps(_build_osworld_tools_def(), ensure_ascii=False)
    date_str = (today or _dt.date.today()).strftime("%A, %B %d, %Y")
    return (
        f"{_OSWORLD_PERSONA}\n\n"
        "# Tools\n\n"
        "You have access to the following functions:\n\n"
        f"<tools>\n{tools_json}\n</tools>\n\n"
        "If you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
        "<tool_call>\n"
        "<function=example_function_name>\n"
        "<parameter=example_parameter_1>\n"
        "value_1\n"
        "</parameter>\n"
        "<parameter=example_parameter_2>\n"
        "This is the value for the second parameter\n"
        "that can span\n"
        "multiple lines\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>\n\n"
        "<IMPORTANT>\n"
        "Reminder:\n"
        "- Function calls MUST follow the specified format: an inner <function=...></function> "
        "block must be nested within <tool_call></tool_call> XML tags\n"
        "- Required parameters MUST be specified\n"
        "- You may provide optional reasoning for your function call in natural language "
        "BEFORE the function call, but NOT after\n"
        "- If there is no function call available, answer the question like normal with your "
        "current knowledge and do not tell the user about function calls\n"
        f"- The current date is {date_str}.\n"
        f"- Collapsed screenshots appear as text: {OSWORLD_COLLAPSE_TEXT}\n"
        "</IMPORTANT>\n\n"
        "# Response format\n\n"
        "Response format for every step:\n"
        "1) Action: a short imperative describing what to do in the UI.\n"
        "2) A single <tool_call>...</tool_call> block.\n\n"
        "Rules:\n"
        "- Output exactly in the order: Action, <tool_call>.\n"
        "- Be brief: one sentence for Action.\n"
        "- Do not output anything else outside those parts.\n"
        "- If finishing, use action=terminate in the tool call."
    )


def _wrap_tool_response(text: str) -> str:
    """Match ``Qwen35VLAgent._wrap_tool_response``: the agent wraps the content
    *inside* a user message (role stays ``user``)."""
    return f"<tool_response>\n{text}\n</tool_response>"


def _format_assistant_body(thought: str, action_nl: str) -> str:
    """Build the ``<think>…</think>\\n\\nAction: {action_nl}\\n`` prefix that
    precedes ``<tool_call>`` in the assistant turn, matching the agent's
    advertised ``# Response format`` (Action: imperative, then tool call)
    while retaining ``<think>`` CoT supervision."""
    thought = (thought or "").strip()
    action_nl = (action_nl or "").strip()
    parts: list[str] = []
    if thought:
        parts.append(f"<think>\n{_escape_image_tag(thought)}\n</think>")
    if action_nl:
        parts.append(f"Action: {_escape_image_tag(action_nl)}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n"


def build_agentnet_sample(
    spec: AgentNetSpec,
    image_history_n: int,
    system_prompt: str | None = None,
) -> dict[str, Any] | None:
    """Build one sharegpt sample for an AgentNet trajectory, aligned with the
    OSWorld PR #448 inference format.

    Message layout:
      * ``role="system"``  → full OSWorld system prompt (persona + tools +
        IMPORTANT + response-format). Shared across all samples when
        ``system_prompt`` is passed in; otherwise computed per-call with
        today's date baked in.
      * ``role="user"``    (turn 0): ``<image>\\n<instruction_prompt>``
      * ``role="function_call"`` (turn 0): ``<think>…</think>\\n\\n
        Action: …\\n<tool_call>{JSON}</tool_call>``
      * ``role="user"``    (turn i>0): ``<tool_response>\\n<image>\\n
        </tool_response>`` — matching the agent's subsequent-turn wrapping.
      * …alternating until EOS.

    ``tools`` is emitted as ``""`` because the tool schema is already baked
    into the system message; leaving it empty prevents LLaMA-Factory's
    ``qwen3_5`` template from appending a second tool prompt.
    """
    if not spec.steps:
        return None

    n = len(spec.steps)
    keep_from_idx = max(0, n - image_history_n - 1)

    messages: list[dict[str, str]] = []
    images: list[str] = []

    if system_prompt is None:
        system_prompt = _build_osworld_system_prompt()
    messages.append({"role": "system", "content": system_prompt})

    first_instruction_text = OSWORLD_INSTRUCTION_PROMPT_TMPL.format(
        instruction=_escape_image_tag(spec.instruction)
    )

    for i, step in enumerate(spec.steps):
        calls = parse_agentnet_code(step.code)

        # ---- user turn: screenshot (or collapse text) ----
        if i >= keep_from_idx:
            img_payload = IMAGE_PLACEHOLDER
            images.append(step.image_path)
        else:
            img_payload = OSWORLD_COLLAPSE_TEXT

        if i == 0:
            user_content = f"{img_payload}\n{first_instruction_text}"
        else:
            user_content = _wrap_tool_response(img_payload)
        messages.append({"role": "user", "content": user_content})

        # ---- assistant/function turn ----
        body = _format_assistant_body(step.thought, step.action_nl)
        if calls:
            payload = calls[0] if len(calls) == 1 else calls
            tool_call = _escape_image_tag(json.dumps(payload, ensure_ascii=False))
            content = f"{body}<tool_call>{tool_call}</tool_call>"
            messages.append({"role": "function_call", "content": content})
        else:
            # Unparsable / no-op step: keep the turn so alternation holds but
            # don't emit a bogus tool call.
            messages.append({"role": "assistant", "content": body})

    # Drop trajectories where we emitted no parsable actions at all.
    if not any(m["role"] == "function_call" for m in messages):
        return None

    # tools left empty on purpose — the schema is embedded in the system msg.
    return {
        "messages": messages,
        "images": images,
        "tools": "",
        "task_id": spec.task_id,
    }


# -----------------------------------------------------------------------------
# Multiprocessing entry
# -----------------------------------------------------------------------------

_WORKER_CTX: dict[str, Any] = {}


def _worker_init(image_history_n: int, system_prompt: str) -> None:
    _WORKER_CTX["image_history_n"] = image_history_n
    _WORKER_CTX["system_prompt"] = system_prompt


def _worker_run(spec: AgentNetSpec) -> dict[str, Any] | None:
    try:
        return build_agentnet_sample(
            spec,
            _WORKER_CTX["image_history_n"],
            system_prompt=_WORKER_CTX["system_prompt"],
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[agentnet] skipped task={spec.task_id}: {exc}")
        return None


def convert_agentnet_specs(
    specs: list[AgentNetSpec],
    image_history_n: int = 3,
    num_workers: int = 32,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Freeze the system prompt once for the whole run so the baked date
    string is identical across every sample (otherwise workers spawned at
    midnight could produce two different dates in the same parquet)."""
    if system_prompt is None:
        system_prompt = _build_osworld_system_prompt()
    if num_workers <= 1:
        _worker_init(image_history_n, system_prompt)
        return [s for s in (_worker_run(sp) for sp in specs) if s is not None]
    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(image_history_n, system_prompt),
    ) as pool:
        out = pool.map(_worker_run, specs, chunksize=max(1, len(specs) // (num_workers * 4)))
    return [s for s in out if s is not None]
