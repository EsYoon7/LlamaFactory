"""Web-agent computer-use SFT dataset converter.

Converts (task_id, step_index, action_index)-granular parquet rows from the
web-agent trajectory dataset into LLaMA-Factory sharegpt-format samples
that use Qwen3-VL's ComputerUse tool schema.

Public entry point: :func:`build_sample` for a single trajectory, or
:func:`convert_dataframe` for a whole DataFrame with multiprocessing.

The raw data has:
  * one row per (task_id, step_index, action_index)
  * shared `thought` and `all_actions` across rows of the same step
  * image paths under a separate blob container (caller must resolve to
    local paths before calling this module)

Design notes:
  * We fix Qwen3.5 tool-call format (XML-nested) via LLaMA-Factory's
    ``qwen3_5`` template; messages are emitted as sharegpt with
    ``role="function_call"`` whose content is JSON ``[{name,arguments},...]``
    for parallel actions (LLaMA-Factory's ``FunctionFormatter`` parses this
    and invokes ``Qwen35ToolUtils.function_formatter``).
  * Reasoning is wrapped in ``<think>...</think>`` before the JSON so that
    ``FunctionFormatter`` preserves it in the final rendered assistant
    text. For steps with no actions, we emit an ``assistant`` turn with
    the ``<think>...</think>`` reasoning only.
  * Image-history trimming keeps only the last ``image_history_n``
    screenshots (plus the current-step screenshot, which is always kept).
    Replaced screenshots become a plain ``[previous screenshot omitted]``
    text placeholder.
  * Coordinate rescaling targets the fixed ``DISPLAY_W × DISPLAY_H``
    normalized space (1000×1000) that Qwen3-VL's computer_use tool is
    trained on — decoupled from the actual image resolution. Inference
    pipelines map back to pixels via ``coord / DISPLAY_* * resized_*``
    (see Qwen3-VL cookbook ``computer_use.ipynb``).

``action_split_mode``:
  * ``all_at_once`` (implemented): every step becomes a single assistant
    turn that emits all actions of the step in one parallel function-call.
  * ``split_per_action`` (skeleton only): each action becomes its own
    turn. Interface is spec'd below; body is NotImplementedError pending
    design decisions (see TODO).

``action_mapping_mode``:
  * ``extend_schema``: augment the ComputerUse tool schema with new
    actions (``visit_url``, ``find_on_page``, ``web_search``,
    ``history_back``, ``page_recovery``) so the dataset's full action
    vocabulary is covered. Adds a ``url`` parameter.
  * ``decompose``: map unsupported actions into sequences of primitives
    from the stock ComputerUse schema (e.g. ``visit_url`` →
    ``key(ctrl+l) + type(url) + key(Enter)``). Loses fidelity but keeps
    the tool schema minimal.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import Any

# -----------------------------------------------------------------------------
# ComputerUse tool schema (verbatim from Qwen3-VL cookbook / agent_function_call.py)
# -----------------------------------------------------------------------------

# Qwen3-VL computer_use convention: model I/O coordinates live in a fixed
# 0..DISPLAY_W × 0..DISPLAY_H normalized space, independent of the actual
# image resolution. Inference-time mapping to pixels is done by callers as
# `coord / DISPLAY_* * resized_*` (see Qwen3-VL cookbook computer_use.ipynb).
# Tool schema display fields and training-label rescaling share this constant.
DISPLAY_W: int = 1000
DISPLAY_H: int = 1000

_COMPUTER_USE_DESCRIPTION = (
    "Use a mouse and keyboard to interact with a computer, and take screenshots.\n"
    "* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. "
    "You must click on desktop icons to start applications.\n"
    "* Some applications may take time to start or process actions, so you may need to wait and take "
    "successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window "
    "doesn't open, try wait and taking another screenshot.\n"
    "* The screen's resolution is {w}x{h}.\n"
    "* Whenever you intend to move the cursor to click on an element like an icon, you should consult a "
    "screenshot to determine the coordinates of the element before moving the cursor.\n"
    "* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting "
    "your cursor position so that the tip of the cursor visually falls on the element that you want to click.\n"
    "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
    "Don't click boxes on their edges."
)

_BASE_ACTIONS = [
    "key", "type", "mouse_move", "left_click", "left_click_drag", "right_click",
    "middle_click", "double_click", "triple_click", "scroll", "hscroll",
    "wait", "terminate", "answer",
]

_EXTENDED_ACTIONS = ["visit_url", "find_on_page", "web_search", "history_back", "history_forward", "page_recovery"]

_BASE_ACTION_DESC = (
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

_EXTENDED_ACTION_EXTRA_DESC = (
    "\n* `visit_url`: Navigate the browser to the given URL.\n"
    "* `find_on_page`: Search for a string within the current page (Ctrl/Cmd+F style).\n"
    "* `web_search`: Issue a web search query and open the results page.\n"
    "* `history_back`: Go back to the previous page in browser history.\n"
    "* `history_forward`: Go forward to the next page in browser history.\n"
    "* `page_recovery`: Attempt to recover from a stuck or broken page state."
)


def build_tool_schema(action_mapping_mode: str, display_w: int = DISPLAY_W, display_h: int = DISPLAY_H) -> list[dict[str, Any]]:
    """Return the ``tools`` list to be JSON-serialized into the sample's ``tools`` field.

    Qwen3.5 template (``Qwen35ToolUtils.tool_formatter``) expects entries that
    are *function* dicts (it unwraps ``{"type":"function","function":{...}}``
    if present). We emit the unwrapped form directly.
    """
    if action_mapping_mode not in ("extend_schema", "decompose"):
        raise ValueError(f"Unknown action_mapping_mode: {action_mapping_mode}")

    extend = action_mapping_mode == "extend_schema"
    action_desc = _BASE_ACTION_DESC + (_EXTENDED_ACTION_EXTRA_DESC if extend else "")
    action_enum = list(_BASE_ACTIONS) + (_EXTENDED_ACTIONS if extend else [])

    properties = {
        "action": {"description": action_desc, "enum": action_enum, "type": "string"},
        "keys": {"description": "Required only by `action=key`.", "type": "array"},
        "text": {
            "description": "Required only by `action=type`, `action=answer`"
            + (", `action=find_on_page`, and `action=web_search`" if extend else "")
            + ".",
            "type": "string",
        },
        "coordinate": {
            "description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
            "coordinates to move the mouse to.",
            "type": "array",
        },
        "pixels": {
            "description": "The amount of scrolling to perform. Positive values scroll up, negative values scroll down. "
            "Required only by `action=scroll` and `action=hscroll`.",
            "type": "number",
        },
        "time": {"description": "The seconds to wait. Required only by `action=wait`.", "type": "number"},
        "status": {
            "description": "The status of the task. Required only by `action=terminate`.",
            "type": "string",
            "enum": ["success", "failure"],
        },
    }
    if extend:
        properties["url"] = {
            "description": "Required only by `action=visit_url`.",
            "type": "string",
        }

    tool = {
        "name": "computer_use",
        "description": _COMPUTER_USE_DESCRIPTION.format(w=display_w, h=display_h),
        "parameters": {"properties": properties, "required": ["action"], "type": "object"},
    }
    return [tool]


# -----------------------------------------------------------------------------
# smart_resize (ported from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast)
# -----------------------------------------------------------------------------

def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
    """Return (resized_h, resized_w) such that both are multiples of ``factor`` and
    the total pixel count is within [min_pixels, max_pixels].
    """
    if height < factor or width < factor:
        raise ValueError(f"height and width must be >= factor ({factor}), got ({height}, {width})")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    return h_bar, w_bar


# -----------------------------------------------------------------------------
# Action mapping
# -----------------------------------------------------------------------------

def _rescale_xy(x: float, y: float, orig_w: int, orig_h: int, target_w: int, target_h: int) -> list[int]:
    """Rescale original-pixel (x, y) into the target coordinate space (typically DISPLAY_W × DISPLAY_H)."""
    nx = int(round(x * target_w / orig_w))
    ny = int(round(y * target_h / orig_h))
    return [nx, ny]


def map_action(
    action: dict[str, Any],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    action_mapping_mode: str,
) -> list[dict[str, Any]]:
    """Convert one raw dataset action dict into zero or more ComputerUse calls.

    Returns a list of ``{"name": "computer_use", "arguments": {...}}`` dicts.
    Returns an empty list for a pure observation (``type=screenshot``) so the
    caller can decide to emit a reasoning-only turn instead of a tool call.
    """
    t = action.get("type")

    def call(args: dict[str, Any]) -> dict[str, Any]:
        return {"name": "computer_use", "arguments": args}

    if t == "screenshot":
        return []
    if t == "wait":
        return [call({"action": "wait", "time": action.get("time", 1)})]
    if t == "type":
        return [call({"action": "type", "text": action.get("text", "")})]
    if t == "hotkey":
        return [call({"action": "key", "keys": list(action.get("keys", []))})]
    if t == "click":
        return [call({"action": "left_click", "coordinate": _rescale_xy(action["x"], action["y"], orig_w, orig_h, target_w, target_h)})]
    if t == "double_click":
        return [call({"action": "double_click", "coordinate": _rescale_xy(action["x"], action["y"], orig_w, orig_h, target_w, target_h)})]
    if t == "right_click":
        return [call({"action": "right_click", "coordinate": _rescale_xy(action["x"], action["y"], orig_w, orig_h, target_w, target_h)})]
    if t == "middle_click":
        return [call({"action": "middle_click", "coordinate": _rescale_xy(action["x"], action["y"], orig_w, orig_h, target_w, target_h)})]
    if t == "triple_click":
        return [call({"action": "triple_click", "coordinate": _rescale_xy(action["x"], action["y"], orig_w, orig_h, target_w, target_h)})]
    if t == "hover":
        return [call({"action": "mouse_move", "coordinate": _rescale_xy(action["x"], action["y"], orig_w, orig_h, target_w, target_h)})]
    if t == "drag":
        # Dataset drag carries start_x/start_y and end_x/end_y (plus a dense path).
        # left_click_drag in ComputerUse drags from the *current* cursor to the
        # given coordinate, so we first mouse_move to the start then drag to the end.
        sx, sy = action.get("start_x"), action.get("start_y")
        ex, ey = action.get("end_x"), action.get("end_y")
        if ex is None or ey is None:
            # fall back to last path point if end coords missing
            path = action.get("path") or []
            if path:
                ex, ey = path[-1].get("x"), path[-1].get("y")
            else:
                ex, ey = action.get("x"), action.get("y")
        calls: list[dict[str, Any]] = []
        if sx is not None and sy is not None:
            calls.append(call({"action": "mouse_move", "coordinate": _rescale_xy(sx, sy, orig_w, orig_h, target_w, target_h)}))
        calls.append(call({"action": "left_click_drag", "coordinate": _rescale_xy(ex, ey, orig_w, orig_h, target_w, target_h)}))
        return calls
    if t == "scroll":
        # (b) Use Qwen's native scroll/hscroll split. Precede with mouse_move when (x,y) present.
        calls: list[dict[str, Any]] = []
        sx, sy = action.get("scroll_x", 0) or 0, action.get("scroll_y", 0) or 0
        ax, ay = action.get("x"), action.get("y")
        if ax is not None and ay is not None:
            calls.append(call({"action": "mouse_move", "coordinate": _rescale_xy(ax, ay, orig_w, orig_h, target_w, target_h)}))
        if sy != 0:
            # Qwen convention: positive=up. Dataset's sign semantics match (positive scroll_y = down in browsers,
            # but we keep the raw sign; tool schema documents positive=up. If downstream needs the opposite,
            # a future flag can flip it.).
            calls.append(call({"action": "scroll", "pixels": sy}))
        if sx != 0:
            calls.append(call({"action": "hscroll", "pixels": sx}))
        if not calls:
            calls.append(call({"action": "scroll", "pixels": 0}))
        return calls

    # Actions not in the base ComputerUse schema
    if action_mapping_mode == "extend_schema":
        if t == "visit_url":
            return [call({"action": "visit_url", "url": action.get("url", "")})]
        if t == "find_on_page":
            return [call({"action": "find_on_page", "text": action.get("text", "")})]
        if t == "web_search":
            return [call({"action": "web_search", "text": action.get("query", action.get("text", ""))})]
        if t == "history_back":
            return [call({"action": "history_back"})]
        if t == "history_forward":
            return [call({"action": "history_forward"})]
        if t == "PAGE_RECOVERY":
            return [call({"action": "page_recovery"})]
    elif action_mapping_mode == "decompose":
        if t == "visit_url":
            # Ctrl+L (focus address bar) → type URL → Enter
            return [
                call({"action": "key", "keys": ["ctrl", "l"]}),
                call({"action": "type", "text": action.get("url", "")}),
                call({"action": "key", "keys": ["Enter"]}),
            ]
        if t == "find_on_page":
            return [
                call({"action": "key", "keys": ["ctrl", "f"]}),
                call({"action": "type", "text": action.get("text", "")}),
                call({"action": "key", "keys": ["Enter"]}),
            ]
        if t == "web_search":
            # Treat as visit_url to google search.
            q = action.get("query", action.get("text", ""))
            return [
                call({"action": "key", "keys": ["ctrl", "l"]}),
                call({"action": "type", "text": f"https://www.google.com/search?q={q}"}),
                call({"action": "key", "keys": ["Enter"]}),
            ]
        if t == "history_back":
            return [call({"action": "key", "keys": ["alt", "Left"]})]
        if t == "history_forward":
            return [call({"action": "key", "keys": ["alt", "Right"]})]
        if t == "PAGE_RECOVERY":
            return [call({"action": "key", "keys": ["F5"]})]
    # terminate / answer pass-through (dataset may not emit these but we accept them)
    if t == "terminate":
        return [call({"action": "terminate", "status": action.get("status", "success")})]
    if t == "answer":
        return [call({"action": "answer", "text": action.get("text", "")})]

    raise ValueError(f"Unhandled action type under mode={action_mapping_mode}: {action}")


# -----------------------------------------------------------------------------
# Trajectory sample construction
# -----------------------------------------------------------------------------

IMAGE_PLACEHOLDER = "<image>"
IMAGE_OMITTED_TEXT = "[previous screenshot omitted]"


def _escape_image_tag(s: str) -> str:
    """Neutralise literal ``<image>`` / ``<video>`` / ``<audio>`` substrings in data text.

    LLaMA-Factory's MM plugin counts ``<image>`` occurrences in message content
    to pair them with the ``images`` list. Trajectory ``instruction`` / task
    ``answer`` / ``thought`` fields may organically contain these substrings
    (e.g. RSS/XML snippets like ``<image><url>...``), which would corrupt the
    pairing. We rename them so they parse identically to a human but are
    invisible to the placeholder counter.
    """
    if not s:
        return s
    return (
        s.replace("<image>", "<image_tag>")
         .replace("<video>", "<video_tag>")
         .replace("<audio>", "<audio_tag>")
    )


@dataclass
class StepInfo:
    step_index: int
    thought: str
    all_actions: list[dict[str, Any]]
    image_path: str          # local path to the `before_image` of this step
    image_w: int
    image_h: int
    answer: str | None = None    # final answer string (only populated for terminal step)


@dataclass
class TrajectorySpec:
    task_id: str
    instruction: str
    steps: list[StepInfo]
    # Deprecated: coordinates are now rescaled to a fixed DISPLAY_W × DISPLAY_H
    # space (Qwen3-VL computer_use convention), not to the smart_resize output.
    # Retained for call-site backward compatibility.
    smart_resize_factor: int = 32
    smart_resize_min_pixels: int = 56 * 56
    smart_resize_max_pixels: int = 14 * 14 * 4 * 1280


def _format_reasoning(thought: str) -> str:
    """Wrap reasoning in ``<think>\\n...\\n</think>\\n\\n`` (the exact token pair the
    qwen3_5 template declares via ``thought_words``) so that downstream
    ReasoningTemplate extracts it cleanly.
    """
    if not thought:
        return ""
    return f"<think>\n{_escape_image_tag(thought.strip())}\n</think>\n\n"


def _build_assistant_content_with_calls(thought: str, calls: list[dict[str, Any]]) -> str:
    """Produce the content string for a sharegpt ``function_call`` turn.

    Layout: ``<think>…</think>\\n\\n<tool_call>[{...}, ...]</tool_call>``.
    LLaMA-Factory's ``FunctionFormatter`` with ``tool_call_words`` matches the
    ``<tool_call>…</tool_call>`` block, JSON-parses the inside, and keeps the
    ``<think>…</think>`` prefix verbatim in the rendered assistant string.
    """
    reasoning = _format_reasoning(thought)
    payload = json.dumps(calls[0] if len(calls) == 1 else calls, ensure_ascii=False)
    # Neutralise any stray <image>/<video>/<audio> substring inside the JSON
    # (e.g. a ``type`` action whose text happens to contain ``<image>``).
    payload = _escape_image_tag(payload)
    return f"{reasoning}<tool_call>{payload}</tool_call>"


def build_sample_all_at_once(spec: TrajectorySpec, image_history_n: int, action_mapping_mode: str) -> dict[str, Any]:
    """Emit one sharegpt sample for a trajectory in ``all_at_once`` mode.

    Layout (strict odd/even alternation required by LLaMA-Factory):
      * turn 0 (user):       ``<image>\\n<instruction>``
      * turn 1 (assistant/function_call): step 0 output
      * turn 2 (user):       ``<image>`` for step 1's before_image
      * turn 3:              step 1 output
      * ...

    Image history trimming: only the last ``image_history_n`` past screenshots
    (not counting the current step) are kept; older ones are replaced by a
    text placeholder. ``image_history_n==0`` means only the current-step
    screenshot is shown.
    """
    if not spec.steps:
        raise ValueError("empty trajectory")

    n_steps = len(spec.steps)
    # Determine which step indices (by position, 0..n_steps-1) keep their images.
    # The "current" step's image is always kept when building its user turn. When
    # we view the finalized trajectory, the LAST step is the "latest current" —
    # so effectively the kept set is the last (image_history_n + 1) steps.
    keep_from_idx = max(0, n_steps - image_history_n - 1)

    messages: list[dict[str, str]] = []
    images: list[str] = []

    for i, step in enumerate(spec.steps):
        # ----- user turn (screenshot + optional instruction) -----
        if i >= keep_from_idx:
            img_token = IMAGE_PLACEHOLDER
            images.append(step.image_path)
        else:
            img_token = IMAGE_OMITTED_TEXT
        if i == 0:
            user_content = f"{img_token}\nTask: {_escape_image_tag(spec.instruction)}"
        else:
            user_content = img_token
        messages.append({"role": "user", "content": user_content})

        # ----- assistant/function turn -----
        # Coordinates are rescaled to the fixed DISPLAY_W × DISPLAY_H normalized
        # space that Qwen3-VL's computer_use tool operates in; this is decoupled
        # from whatever image resolution the mm_plugin feeds to the model.
        calls: list[dict[str, Any]] = []
        for raw_action in step.all_actions:
            calls.extend(map_action(raw_action, step.image_w, step.image_h, DISPLAY_W, DISPLAY_H, action_mapping_mode))
        if calls:
            messages.append({
                "role": "function_call",
                "content": _build_assistant_content_with_calls(step.thought, calls),
            })
        else:
            # Pure observation step: reasoning-only assistant turn. Use the
            # same <think>\n...\n</think>\n\n wrapper so ReasoningTemplate parses
            # it consistently. If no reasoning, fall back to empty string.
            messages.append({"role": "assistant", "content": _format_reasoning(step.thought)})

    # Attach final answer as a trailing assistant turn if present and not already
    # emitted (keeps the ``answer`` field of the task visible as the terminal
    # assistant message). We add a dummy user turn to preserve alternation.
    if spec.steps[-1].answer:
        messages.append({"role": "user", "content": "Please provide the final answer."})
        safe_answer = _escape_image_tag(spec.steps[-1].answer)
        messages.append({
            "role": "function_call",
            "content": "<tool_call>" + json.dumps(
                {"name": "computer_use", "arguments": {"action": "answer", "text": safe_answer}},
                ensure_ascii=False,
            ) + "</tool_call>",
        })

    tools_str = json.dumps(build_tool_schema(action_mapping_mode), ensure_ascii=False)

    return {
        "messages": messages,
        "images": images,
        "tools": tools_str,
        "task_id": spec.task_id,
    }


# -----------------------------------------------------------------------------
# TODO: split_per_action mode
# -----------------------------------------------------------------------------

def build_sample_split_per_action(spec: TrajectorySpec, image_history_n: int, action_mapping_mode: str) -> dict[str, Any]:
    """Emit one sharegpt sample for a trajectory in ``split_per_action`` mode.

    TODO: design not finalized. Intended semantics:
      * Each action in a multi-action step becomes its own assistant turn.
      * Input per sub-turn: same underlying screenshot + cumulative action
        history (previously predicted actions of this step appended as prior
        assistant turns, each with their ``tool_response``-less observation
        being "(no environment update between sub-actions)").
      * Output per sub-turn: exactly one ``computer_use`` call.
      * Reasoning handling (two candidates, pick one when implementing):
          (a) drop reasoning on intra-step sub-turns (only the first sub-turn
              of each step carries the step's reasoning, the rest are pure
              action predictions).
          (b) inject a synthetic reasoning such as
              "Following the reasoning from the previous sub-step, the next
              sub-action is ...".
    The input/output schema (sharegpt messages + images + tools) is the same
    as :func:`build_sample_all_at_once`; only the turn-splitting changes.
    """
    raise NotImplementedError(
        "split_per_action mode is not yet implemented. See TODO in web_agent_cu.py."
    )


# -----------------------------------------------------------------------------
# Top-level conversion (multi-processing friendly)
# -----------------------------------------------------------------------------

_WORKER_CTX: dict[str, Any] = {}


def _worker_init(image_history_n: int, action_split_mode: str, action_mapping_mode: str) -> None:
    _WORKER_CTX["image_history_n"] = image_history_n
    _WORKER_CTX["action_split_mode"] = action_split_mode
    _WORKER_CTX["action_mapping_mode"] = action_mapping_mode


def _worker_run(spec: TrajectorySpec) -> dict[str, Any] | None:
    mode = _WORKER_CTX["action_split_mode"]
    try:
        if mode == "all_at_once":
            return build_sample_all_at_once(spec, _WORKER_CTX["image_history_n"], _WORKER_CTX["action_mapping_mode"])
        elif mode == "split_per_action":
            return build_sample_split_per_action(spec, _WORKER_CTX["image_history_n"], _WORKER_CTX["action_mapping_mode"])
        else:
            raise ValueError(f"Unknown action_split_mode: {mode}")
    except Exception as exc:  # pragma: no cover - defensive
        import traceback
        traceback.print_exc()
        print(f"[web_agent_cu] skipped task={spec.task_id}: {exc}")
        return None


def convert_specs(
    specs: list[TrajectorySpec],
    image_history_n: int = 3,
    action_split_mode: str = "all_at_once",
    action_mapping_mode: str = "extend_schema",
    num_workers: int = 64,
) -> list[dict[str, Any]]:
    """Convert a list of TrajectorySpec → sharegpt samples using a process pool."""
    if num_workers <= 1:
        _worker_init(image_history_n, action_split_mode, action_mapping_mode)
        return [s for s in (_worker_run(spec) for spec in specs) if s is not None]

    with Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(image_history_n, action_split_mode, action_mapping_mode),
    ) as pool:
        out = pool.map(_worker_run, specs, chunksize=max(1, len(specs) // (num_workers * 4)))
    return [s for s in out if s is not None]
