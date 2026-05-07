"""Tool-using agent loop against an OpenAI-compatible vLLM endpoint.

Loop:
  1. Send messages + tools (search, fetch_url) to the served model.
  2. If model returns tool_calls -> execute, append tool result, go to 1.
  3. Else return final assistant content.

Tools follow OpenAI function-calling JSON schema.
"""
from __future__ import annotations
import json, os, time
from openai import OpenAI
from search_tool import search, fetch_url


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Run a web search query and return the top organic results (title, snippet, link).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "k":      {"type": "integer", "description": "How many results", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the plain text content of a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a web search tool and a URL fetcher. "
    "When you need up-to-date or factual information, call `search` to find candidate sources, "
    "then optionally call `fetch_url` for any link you want to read in depth. "
    "After gathering enough evidence, give a concise final answer. "
    "Always end with a line: \"Final answer: <your concise answer>\"."
)


def run_tool(name: str, args: dict) -> str:
    if name == "search":
        res = search(args.get("query", ""), int(args.get("k", 5) or 5))
        return json.dumps(res)[:8000]
    if name == "fetch_url":
        return fetch_url(args.get("url", ""))[:8000]
    return f"[unknown tool: {name}]"


def answer_with_tools(client: OpenAI, model: str, question: str,
                       max_steps: int = 6, temperature: float = 0.6,
                       max_tokens_per_step: int = 2048) -> dict:
    """Run a tool-use loop. Returns dict with final 'response' and 'trace'."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    trace = []
    for step in range(max_steps):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=temperature,
                max_tokens=max_tokens_per_step,
                timeout=180,
            )
        except Exception as e:
            trace.append({"step": step, "error": f"{type(e).__name__}: {e}"})
            break
        msg = r.choices[0].message
        trace.append({"step": step, "role": "assistant",
                      "content": msg.content,
                      "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})
        if not msg.tool_calls:
            return {"response": msg.content or "", "trace": trace, "stop_reason": r.choices[0].finish_reason}
        # Append assistant turn (with tool_calls) so the model can refer back
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except Exception:
                args = {}
            result = run_tool(tc.function.name, args)
            trace.append({"step": step, "tool": tc.function.name, "args": args, "result": result[:500]})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    # max_steps reached → ask one final answer without tools
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages + [{"role": "user", "content": "Provide your final answer now."}],
            temperature=temperature,
            max_tokens=max_tokens_per_step,
            timeout=180,
        )
        return {"response": r.choices[0].message.content or "", "trace": trace, "stop_reason": "max_steps"}
    except Exception as e:
        return {"response": "", "trace": trace + [{"error": str(e)}], "stop_reason": "error"}
