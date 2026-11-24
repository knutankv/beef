"""Local HTTP server (standard library only) serving the BEEF GUI and its JSON API."""
import json
import mimetypes
import secrets
import shutil
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from .session import JobFailed, Session, UserError

STATIC_DIR = Path(__file__).parent / "static"
API = {}


def api(name):
    """Register an API endpoint: fn(session, params) -> JSON-able data."""
    def deco(fn):
        API[name] = fn
        return fn
    return deco


class _Handler(BaseHTTPRequestHandler):
    server_version = "beef-gui"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if self.server.verbose:
            super().log_message(fmt, *args)

    def _host_ok(self):
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in self.server.allowed_hosts

    def _token_ok(self, query=None):
        tok = self.headers.get("X-Token")
        if tok is None and query is not None:
            tok = (query.get("token") or [None])[0]
        return tok is not None and secrets.compare_digest(tok, self.server.token)

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, allow_nan=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, ctype=None):
        path = Path(path)
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        with open(path, "rb") as f:
            shutil.copyfileobj(f, self.wfile)

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, {"ok": False, "error": "Host not allowed"})
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/static/"):
            target = (STATIC_DIR / path[len("/static/"):]).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                return self._send(404, "not found", "text/plain")
            ctype = "application/javascript" if target.suffix == ".js" else None
            return self._send_file(target, ctype)
        return self._send(404, "not found", "text/plain")

    def do_POST(self):
        if not self._host_ok():
            return self._send(403, {"ok": False, "error": "Host not allowed"})
        url = urlparse(self.path)
        if not self._token_ok(parse_qs(url.query)):
            return self._send(403, {"ok": False, "error": "Invalid or missing session token. "
                                    "Open the URL printed in the terminal (it contains ?token=...)."})
        if not url.path.startswith("/api/"):
            return self._send(404, {"ok": False, "error": "unknown endpoint"})
        name = url.path[len("/api/"):]
        fn = API.get(name)
        if fn is None:
            return self._send(404, {"ok": False, "error": f"Unknown API endpoint {name!r}"})
        try:
            length = int(self.headers.get("Content-Length") or 0)
            params = json.loads(self.rfile.read(length) or b"{}") if length else {}
        except Exception as exc:
            return self._send(400, {"ok": False, "error": f"Bad JSON: {exc}"})
        try:
            data = fn(self.server.session, params)
            try:
                body = json.dumps({"ok": True, "data": data}, allow_nan=False)
            except ValueError:
                from .serialize import jsonable
                body = json.dumps({"ok": True, "data": jsonable(data)}, allow_nan=False)
            return self._send(200, body)
        except UserError as exc:
            return self._send(200, {"ok": False, "error": str(exc)})
        except JobFailed as exc:
            return self._send(200, {"ok": False, "error": exc.job.error, "traceback": exc.job.traceback,
                                    "job": exc.job.id, "log": exc.job.log.text()})
        except Exception as exc:
            return self._send(200, {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()})


class GuiServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, session, token, allowed_hosts, verbose=False):
        super().__init__(addr, _Handler)
        self.session = session
        self.token = token
        self.allowed_hosts = allowed_hosts
        self.verbose = verbose


def make_server(host="127.0.0.1", port=8070, token=None, verbose=False, allow_hosts=()):
    """Create (but do not start) the server. Returns (httpd, url)."""
    from . import api as _api  # noqa: F401  (registers endpoints)
    from .bootstrap import build_namespace

    session = Session()
    session.init_namespace(build_namespace())
    token = token or secrets.token_urlsafe(18)
    allowed = {"localhost", "127.0.0.1", "::1"} | {h.lower() for h in allow_hosts}
    if host not in ("127.0.0.1", "localhost", "::1", ""):
        allowed.add(host.lower())
    httpd = None
    for p in range(port, port + 20):
        try:
            httpd = GuiServer((host, p), session, token, allowed, verbose)
            port = p
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit(f"Could not bind to any port in {port}-{port + 19}")
    # 127.0.0.1 rather than "localhost": on Windows "localhost" resolves to ::1 first, which we do not bind
    shown_host = "127.0.0.1" if host in ("127.0.0.1", "", "0.0.0.0") else host
    return httpd, f"http://{shown_host}:{port}/?token={token}"


def serve(host="127.0.0.1", port=8070, open_browser=True, token=None, verbose=False, allow_hosts=()):
    httpd, url = make_server(host, port, token, verbose, allow_hosts)
    import beef
    print(f"\n  BEEF GUI is running (beef {beef.__version__} from {Path(beef.__file__).parent})")
    print(f"  Open: {url}\n  (Ctrl+C to stop)\n", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping BEEF GUI.")
    finally:
        httpd.server_close()
    return url
