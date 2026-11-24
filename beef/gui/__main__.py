import argparse

from . import __version__


def main(argv=None):
    ap = argparse.ArgumentParser(prog="beef-gui", description="Browser-based GUI for beef (BEam Elements Framework)")
    ap.add_argument("--host", default="127.0.0.1", help="Interface to bind (default 127.0.0.1 = this computer only)")
    ap.add_argument("--port", type=int, default=8070, help="Port (default 8070; next free port is used if busy)")
    ap.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    ap.add_argument("--token", default=None, help="Fixed session token (default: random)")
    ap.add_argument("--allow-host", action="append", default=[], help="Extra Host header value to accept")
    ap.add_argument("--verbose", action="store_true", help="Log every HTTP request")
    ap.add_argument("--version", action="version", version=f"beef-gui {__version__}")
    a = ap.parse_args(argv)
    from .server import serve
    serve(host=a.host, port=a.port, open_browser=not a.no_browser, token=a.token, verbose=a.verbose,
          allow_hosts=a.allow_host)


if __name__ == "__main__":
    main()
