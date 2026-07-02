"""Turn a :class:`~dashboard.script_specs.ScriptSpec` into a command, and (in
Streamlit) into a form.

``build_command`` is a pure function -- given a spec and a values dict it returns
the exact argv the CLI would take -- and is the piece the tests exercise.
``render_form`` is the thin Streamlit layer that gathers those values from
widgets and calls ``build_command``.
"""
import sys
from typing import Any, Dict, List

from script_specs import ArgSpec, ScriptSpec


def _tokens_for(arg: ArgSpec, value: Any) -> List[str]:
    """The argv fragment a single arg contributes (possibly empty)."""
    if arg.kind == "bool":
        return [arg.flag] if value else []

    if arg.kind == "paths":
        if isinstance(value, str):
            items = value.split()
        else:
            items = [str(item).strip() for item in (value or []) if str(item).strip()]
        if not items:
            if arg.required:
                raise ValueError(f"{arg.flag} is required")
            return []
        return [arg.flag, *items]

    # scalar kinds: int / float / str / choice / path
    is_blank = value is None or (isinstance(value, str) and value.strip() == "")
    if is_blank:
        if arg.required:
            raise ValueError(f"{arg.flag} is required")
        return []
    text = str(value).strip() if isinstance(value, str) else str(value)
    return [arg.flag, text]


def build_command(spec: ScriptSpec, values: Dict[str, Any]) -> List[str]:
    """Assemble the full argv for ``spec`` from a ``{arg.name: value}`` dict.

    Uses the current interpreter (``sys.executable``) so the dashboard's venv is
    the one that runs the script. Raises ``ValueError`` if a required arg is blank.
    """
    argv: List[str] = [sys.executable, spec.script]
    if spec.subcommand:
        argv.append(spec.subcommand)
    for arg in spec.args:
        value = values.get(arg.name, arg.default)
        argv.extend(_tokens_for(arg, value))
    return argv


def render_form(spec: ScriptSpec, key_prefix: str = "") -> List[str]:
    """Render a Streamlit form body for ``spec`` and return the built argv.

    Imported lazily so ``build_command`` stays importable without Streamlit.
    """
    import streamlit as st

    prefix = key_prefix or f"{spec.script}:{spec.subcommand or spec.key}"
    values: Dict[str, Any] = {}
    for arg in spec.args:
        widget_key = f"{prefix}:{arg.name}"
        label = arg.label or arg.flag
        if arg.kind == "int":
            values[arg.name] = int(
                st.number_input(label, value=int(arg.default or 0), step=1, key=widget_key)
            )
        elif arg.kind == "float":
            values[arg.name] = float(
                st.number_input(
                    label, value=float(arg.default or 0.0), step=0.01,
                    format="%.4f", key=widget_key,
                )
            )
        elif arg.kind == "bool":
            values[arg.name] = st.checkbox(label, value=bool(arg.default), key=widget_key)
        elif arg.kind == "choice":
            options = list(arg.choices)
            if not arg.required and "" not in options:
                options = ["(default)"] + options
            default = arg.default if arg.default in options else options[0]
            picked = st.selectbox(label, options, index=options.index(default), key=widget_key)
            values[arg.name] = "" if picked in ("(default)", "") else picked
        elif arg.kind == "paths":
            raw = st.text_area(label, value="", key=widget_key,
                               placeholder="one path / glob / folder per line")
            values[arg.name] = raw
        else:  # str, path
            values[arg.name] = st.text_input(label, value=str(arg.default or ""), key=widget_key)
        # Show the parameter's description inline (always visible, not hover-only).
        if arg.help:
            required_tag = " · **required**" if arg.required else ""
            st.caption(arg.help + required_tag)
    return build_command(spec, values)
