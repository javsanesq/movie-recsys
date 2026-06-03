"""
Tiny stdlib-only dev server for local testing.
Serves ui/ statically and reverse-proxies API routes to uvicorn on :8000.
Usage: python scripts/dev_serve.py   (binds 0.0.0.0:8080)
"""

import http.server
import os
import urllib.request
import urllib.error
from pathlib import Path

UI_DIR   = Path(__file__).parent.parent / "ui"
UPSTREAM = os.environ.get("API_UPSTREAM", "http://localhost:8000")
API_PREFIXES = ("/health", "/recommend", "/users", "/movies", "/metrics")
PORT = int(os.environ.get("PORT", "8080"))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self):
        if any(self.path.startswith(p) for p in API_PREFIXES):
            self._proxy()
        else:
            super().do_GET()

    def _proxy(self):
        url = UPSTREAM + self.path
        try:
            with urllib.request.urlopen(url) as resp:
                body = resp.read()
                self.send_response(resp.status)
                for key, val in resp.headers.items():
                    if key.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(body)
        except urllib.error.HTTPError as e:
            body = e.read()
            self.send_response(e.code)
            for key, val in e.headers.items():
                if key.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            msg = f"Proxy error: {e}".encode()
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(msg)

    def log_message(self, fmt, *args):  # quieter logs
        print(f"[dev_serve] {self.address_string()} {fmt % args}")


if __name__ == "__main__":
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[dev_serve] Serving ui/ on http://localhost:{PORT}  "
          f"— proxying API to {UPSTREAM}")
    server.serve_forever()
