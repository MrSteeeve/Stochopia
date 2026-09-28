"""A scripted OpenAI-compatible server for testing the plumbing without a model.

    python -m harness.mock_llm --port 8765
    MODEL_ENDPOINT=http://127.0.0.1:8765 MODEL_TOKEN=x ...

Behaviour (deliberately dumb, it exercises every branch of the solve loop):
- the House route contract: only /v1/chat/completions (403 otherwise), Bearer required;
- plan request            -> a small JSON plan;
- exemplar code request   -> a correct Black-Scholes solution;
- other code requests     -> a script that crashes;
- repair requests         -> a script that writes placeholder files for the
                             detected deliverables (fails the output checks);
- review requests         -> "OK".
"""

from __future__ import annotations

import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

EXEMPLAR_SOLUTION = r'''
import os, math
import numpy as np, pandas as pd
from scipy.stats import norm

df = pd.read_parquet("/input/environment/data/options.parquet")
S, K, T, r, sig = (df[c].to_numpy(float) for c in ("S", "K", "T", "r", "sigma"))
d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
d2 = d1 - sig * np.sqrt(T)
call = df["option_type"].to_numpy() == "call"
disc = np.exp(-r * T)
price = np.where(call, S * norm.cdf(d1) - K * disc * norm.cdf(d2), K * disc * norm.cdf(-d2) - S * norm.cdf(-d1))
delta = np.where(call, norm.cdf(d1), norm.cdf(d1) - 1.0)
gamma = norm.pdf(d1) / (S * sig * np.sqrt(T))
vega = S * norm.pdf(d1) * np.sqrt(T)
theta_y = -S * norm.pdf(d1) * sig / (2 * np.sqrt(T)) + np.where(call, -r * K * disc * norm.cdf(d2), r * K * disc * norm.cdf(-d2))
out = pd.DataFrame({"option_id": df["option_id"], "price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta_y / 365.0})
os.makedirs("/output", exist_ok=True)
out.to_parquet("/output/results.parquet", index=False)
print(out.describe())
'''

CRASH = 'raise RuntimeError("mock model: first attempt always crashes")\n'


def placeholder_script(user_text: str) -> str:
    names = re.findall(r'^- \{"name": "([^"]+)"', user_text, re.M)
    lines = ["import json, os", "os.makedirs('/app/output', exist_ok=True)"]
    for n in names:
        path = f"/app/output/{n}"
        if n.endswith(".json"):
            lines.append(f"json.dump({{'placeholder': 1}}, open({path!r}, 'w'))")
        else:
            lines.append(f"open({path!r}, 'w').write('placeholder\\n1\\n')")
    if not names:
        lines.append("open('/app/output/results.json', 'w').write('{}')")
    return "\n".join(lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    calls = 0
    lock = threading.Lock()
    delay_s = 0.0

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            return self._send(403, {"error": "use /v1/chat/completions"})
        if not self.headers.get("Authorization", "").startswith("Bearer ") or self.headers["Authorization"] == "Bearer ":
            return self._send(401, {"error": "missing bearer"})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with Handler.lock:
            Handler.calls += 1
        time.sleep(Handler.delay_s)
        msgs = req.get("messages", [])
        last = msgs[-1]["content"] if msgs else ""
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), "")
        if "return ONLY a JSON object" in last:
            content = json.dumps({"deliverables": [], "method": "mock", "conventions": [], "pitfalls": []})
        elif "reply with exactly OK" in last:
            content = "OK"
        elif "Fix the root cause" in last:
            content = "```python\n" + placeholder_script(first_user) + "```"
        elif "options.parquet" in first_user:
            content = "<think>mock reasoning</think>```python\n" + EXEMPLAR_SOLUTION + "```"
        else:
            content = "```python\n" + CRASH + "```"
        self._send(
            200,
            {
                "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            },
        )


def serve(port: int, delay_s: float = 0.0) -> ThreadingHTTPServer:
    Handler.delay_s = delay_s
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--delay", type=float, default=0.0)
    args = ap.parse_args()
    serve(args.port, args.delay)
    print(f"mock House route on http://127.0.0.1:{args.port} (Ctrl-C to stop)", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
