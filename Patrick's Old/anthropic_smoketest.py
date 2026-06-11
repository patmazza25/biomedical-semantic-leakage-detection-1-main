#!/usr/bin/env python3
# Minimal Anthropic Messages API smoke test for debugging 400 errors.

import os
import sys
import json
import argparse
from typing import Any

def jprint(label: str, obj: Any):
    try:
        print(f"{label}: {json.dumps(obj, ensure_ascii=False, indent=2)[:5000]}")
    except Exception:
        print(f"{label}: {obj}")

def main():
    parser = argparse.ArgumentParser(description="Anthropic Messages API smoke test")
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL_DEFAULT", "claude-haiku-4-5"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--system", default="smoke-test")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    # Basic validations to fail fast before hitting the API.
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY is not set in your environment.", file=sys.stderr)
        sys.exit(2)
    if not isinstance(args.max_tokens, int) or args.max_tokens <= 0:
        print("ERROR: --max-tokens must be a positive integer.", file=sys.stderr)
        sys.exit(2)
    if not args.model.strip():
        print("ERROR: --model must be a non-empty string.", file=sys.stderr)
        sys.exit(2)
    if not isinstance(args.prompt, str) or not args.prompt.strip():
        print("ERROR: --prompt must be a non-empty string.", file=sys.stderr)
        sys.exit(2)

    # Import SDK after env checks so we can show a useful error if missing.
    try:
        import anthropic
        from anthropic import Anthropic
    except Exception as e:
        print("ERROR: Could not import 'anthropic' SDK. Install it with:", file=sys.stderr)
        print("  pip install --upgrade anthropic", file=sys.stderr)
        print(f"Detail: {e}", file=sys.stderr)
        sys.exit(2)

    # Print environment info that often differs across machines.
    print("SDK version:", getattr(anthropic, "__version__", "unknown"))
    print("Model:", args.model)
    print("Max tokens:", args.max_tokens)

    client = Anthropic(timeout=args.timeout)

    payload = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "system": args.system,
        "messages": [
            {"role": "user", "content": args.prompt}
        ],
    }
    jprint("Request payload", payload)

    # Hit the API with strict error surfacing.
    try:
        resp = client.messages.create(**payload)
        # Show minimal successful output.
        text_blocks = [b.text for b in getattr(resp, "content", []) if getattr(b, "type", "") == "text"]
        print("Status: OK")
        if text_blocks:
            print("Reply:", text_blocks[0][:2000])
        else:
            jprint("Raw response", resp.model_dump() if hasattr(resp, "model_dump") else resp)
        sys.exit(0)

    except Exception as e:
        # Anthropic SDK attaches useful attributes; surface anything we can find.
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        detail = None

        # Try common locations for error JSON/body
        for attr in ("response", "body", "error", "message"):
            obj = getattr(e, attr, None)
            if obj:
                detail = obj
                break

        # Some SDK versions put JSON in e.response.json() or e.response.text
        if hasattr(e, "response"):
            resp_obj = getattr(e, "response")
            try:
                if hasattr(resp_obj, "json"):
                    detail = resp_obj.json()
                elif hasattr(resp_obj, "text"):
                    detail = resp_obj.text
            except Exception:
                pass

        print("Status: ERROR")
        print("HTTP status:", status)
        if detail:
            try:
                if isinstance(detail, str):
                    # Attempt to parse JSON string; else print raw.
                    try:
                        jprint("Error body", json.loads(detail))
                    except Exception:
                        print("Error body:", detail[:5000])
                else:
                    jprint("Error body", detail)
            except Exception:
                print("Error detail (unparsed):", str(detail)[:5000])

        # As a last resort, print the exception message
        if not detail:
            print("Error:", str(e)[:5000])

        # Non-zero exit so scripts/CI can fail fast
        sys.exit(1)

if __name__ == "__main__":
    main()
