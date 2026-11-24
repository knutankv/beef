"""Session state: Python namespace, background job queue with stdout capture."""
import ctypes
import io
import itertools
import queue
import re
import sys
import threading
import time
import traceback
from collections import OrderedDict


class JobCancelled(BaseException):
    """Raised asynchronously inside a running job thread when the user cancels it."""


class _ThreadRouter(io.TextIOBase):
    """Routes writes from job threads into the job log, everything else to the original stream."""

    def __init__(self, original):
        self.original = original
        self.targets = {}

    def write(self, s):
        buf = self.targets.get(threading.get_ident())
        if buf is not None:
            buf.write(s)
        try:
            self.original.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self.original.flush()
        except Exception:
            pass

    @property
    def encoding(self):
        return getattr(self.original, "encoding", "utf-8")

    def isatty(self):
        return False


_router_out = None
_router_err = None


def install_routers():
    global _router_out, _router_err
    if _router_out is None:
        _router_out = _ThreadRouter(sys.stdout)
        _router_err = _ThreadRouter(sys.stderr)
        sys.stdout = _router_out
        sys.stderr = _router_err


class _JobLog:
    """Accumulates text, interpreting carriage returns like a terminal."""

    MAX_LINES = 600

    def __init__(self):
        self.lines = [""]
        self.lock = threading.Lock()
        self.version = 0

    def write(self, s):
        with self.lock:
            for part in re.split(r"(\r|\n)", s):
                if part == "\n":
                    self.lines.append("")
                elif part == "\r":
                    self.lines[-1] = "\x00CR"  # marker: next text overwrites the line
                elif part:
                    if self.lines[-1] == "\x00CR":
                        self.lines[-1] = part
                    else:
                        self.lines[-1] += part
            if len(self.lines) > self.MAX_LINES:
                self.lines = self.lines[-self.MAX_LINES:]
            self.version += 1

    def text(self):
        with self.lock:
            return "\n".join(l for l in self.lines if l != "\x00CR")

    def last_line(self):
        with self.lock:
            for l in reversed(self.lines):
                if l and l != "\x00CR":
                    return l
        return ""


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind, label, fn):
        self.id = next(self._ids)
        self.kind = kind
        self.label = label
        self.fn = fn
        self.status = "queued"
        self.log = _JobLog()
        self.result = None
        self.error = None
        self.traceback = None
        self.t_queued = time.time()
        self.t_start = None
        self.t_end = None
        self.thread_ident = None
        self.done_event = threading.Event()

    @property
    def progress(self):
        m = re.findall(r"(\d{1,3})%", self.log.last_line())
        return int(m[-1]) if m else None

    def to_dict(self, with_log=False, with_result=False):
        d = dict(
            id=self.id, kind=self.kind, label=self.label, status=self.status,
            error=self.error, progress=self.progress,
            t_queued=self.t_queued, t_start=self.t_start, t_end=self.t_end,
            elapsed=((self.t_end or time.time()) - self.t_start) if self.t_start else None,
            last_line=self.log.last_line()[-200:],
        )
        if with_log:
            d["log"] = self.log.text()
            d["traceback"] = self.traceback
        if with_result:
            d["result"] = self.result
        return d


class JobManager:
    def __init__(self, max_jobs=200):
        self.jobs = OrderedDict()
        self.q = queue.Queue()
        self.max_jobs = max_jobs
        self.current = None
        self.worker = threading.Thread(target=self._work, name="beef-gui-worker", daemon=True)
        self.worker.start()

    def submit(self, kind, label, fn):
        job = Job(kind, label, fn)
        self.jobs[job.id] = job
        while len(self.jobs) > self.max_jobs:
            oldest = next(iter(self.jobs))
            if self.jobs[oldest].status in ("queued", "running"):
                break
            self.jobs.pop(oldest)
        self.q.put(job)
        return job

    def _work(self):
        while True:
            job = self.q.get()
            if job.status == "cancelled":
                job.done_event.set()
                continue
            self.current = job
            job.status = "running"
            job.t_start = time.time()
            job.thread_ident = threading.get_ident()
            _router_out.targets[job.thread_ident] = job.log
            _router_err.targets[job.thread_ident] = job.log
            try:
                job.result = job.fn()
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
                job.error = "Cancelled by user"
            except UserError as exc:   # message meant for the user: no traceback
                job.status = "error"
                job.error = str(exc)
                job.log.write("\nError: " + str(exc) + "\n")
            except BaseException as exc:  # noqa: B902 - report everything to the UI
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.traceback = traceback.format_exc()
                job.log.write("\n" + job.traceback)
            finally:
                _router_out.targets.pop(job.thread_ident, None)
                _router_err.targets.pop(job.thread_ident, None)
                job.t_end = time.time()
                job.fn = None
                self.current = None
                job.done_event.set()

    def cancel(self, job_id):
        job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(f"No job {job_id}")
        if job.status == "queued":
            job.status = "cancelled"
            job.error = "Cancelled before start"
            return job
        if job.status == "running" and job.thread_ident:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(job.thread_ident), ctypes.py_object(JobCancelled))
        return job

    @property
    def busy(self):
        return self.current is not None or not self.q.empty()


class Session:
    """Holds the Python namespace shared by every panel of the GUI."""

    def __init__(self):
        install_routers()
        self.jobs = JobManager()
        self.ns = {}
        self.base_names = set()
        self.uploads = {}
        self.downloads = {}
        self.mpl_lock = threading.RLock()
        self.meta = {}  # GUI-specific metadata (e.g. source strings of state parameters)

    # ------------------------------------------------------------------ namespace
    def init_namespace(self, extra):
        self.ns.update(extra)
        self.ns["__builtins__"] = __builtins__
        self.base_names = set(self.ns)

    def user_vars(self):
        return {k: v for k, v in self.ns.items()
                if not k.startswith("_") and k not in self.base_names}

    @property
    def model(self):
        return self.ns.get("model")

    def require_model(self):
        m = self.ns.get("model")
        if m is None:
            raise UserError("No model loaded.")
        return m

    # ------------------------------------------------------------------ evaluation
    def ev(self, value, default=None):
        """Evaluate a GUI field: strings are Python expressions in the session namespace."""
        if value is None or (isinstance(value, str) and value.strip() == ""):
            if isinstance(default, str):
                return eval(default, self.ns)
            return default
        if isinstance(value, str):
            s = value.strip()
            try:
                return eval(s, self.ns)
            except SyntaxError as exc:
                raise UserError(f"Invalid expression {s!r}: {exc.msg}")
            except NameError as exc:
                raise UserError(f"Error evaluating {s!r}: {exc}")
        return value

    # ------------------------------------------------------------------ jobs
    def run(self, kind, label, fn, wait=6.0):
        """Queue fn on the worker thread. Returns the result if it completes within `wait`."""
        job = self.jobs.submit(kind, label, fn)
        if wait and job.done_event.wait(wait):
            if job.status == "error":
                raise JobFailed(job)
            if job.status == "cancelled":
                raise UserError("Job cancelled")
            return {"job": job.id, "done": True, "result": job.result, "log": job.log.text()}
        return {"job": job.id, "done": False}


class UserError(Exception):
    """An error with a message meant for the GUI user (no traceback shown)."""


class JobFailed(Exception):
    def __init__(self, job):
        super().__init__(job.error)
        self.job = job
