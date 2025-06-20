import sys
import logging
import time

from concurrent.futures import ThreadPoolExecutor

import celery
import tornado.web

from tornado import ioloop
from tornado.httpserver import HTTPServer
from tornado.web import url
from tornado.ioloop import PeriodicCallback

from .urls import handlers as default_handlers
from .events import Events
from .inspector import Inspector
from .options import default_options


logger = logging.getLogger(__name__)


if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith('win'):
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# pylint: disable=consider-using-f-string
def rewrite_handler(handler, url_prefix):
    if isinstance(handler, url):
        return url("/{}{}".format(url_prefix.strip("/"), handler.regex.pattern),
                   handler.handler_class, handler.kwargs, handler.name)
    return ("/{}{}".format(url_prefix.strip("/"), handler[0]), handler[1])


class Flower(tornado.web.Application):
    pool_executor_cls = ThreadPoolExecutor
    max_workers = None

    def __init__(self, options=None, capp=None, events=None,
                 io_loop=None, **kwargs):
        handlers = default_handlers
        if options is not None and options.url_prefix:
            handlers = [rewrite_handler(h, options.url_prefix) for h in handlers]
        kwargs.update(handlers=handlers)
        super().__init__(**kwargs)
        self.options = options or default_options
        self.io_loop = io_loop or ioloop.IOLoop.instance()
        self.ssl_options = kwargs.get('ssl_options', None)

        self.capp = capp or celery.Celery()
        self.capp.loader.import_default_modules()

        self.executor = self.pool_executor_cls(max_workers=self.max_workers)
        self.io_loop.set_default_executor(self.executor)

        self.inspector = Inspector(self.io_loop, self.capp, self.options.inspect_timeout / 1000.0)

        self.events = events or Events(
            self.capp,
            db=self.options.db,
            persistent=self.options.persistent,
            state_save_interval=self.options.state_save_interval,
            enable_events=self.options.enable_events,
            io_loop=self.io_loop,
            max_workers_in_memory=self.options.max_workers,
            max_tasks_in_memory=self.options.max_tasks)
        self.started = False

    def _log_ioloop(self):
        """Periodic callback that logs basic statistics about the Tornado IOLoop
        and ThreadPoolExecutor queue. This can be very helpful when debugging
        situations where the UI appears to hang because it provides visibility
        into how busy the loop is and how many blocking operations are queued
        in the executor.
        """
        try:
            pending_callbacks = len(getattr(self.io_loop, "_callbacks", []))
            timeouts = len(getattr(self.io_loop, "_timeouts", []))

            # The work queue lives on the executor; accessing a protected member
            # here is acceptable because it is purely for debugging/observability
            # purposes.
            executor_backlog = getattr(self.executor, "_work_queue", None)
            if executor_backlog is not None:
                executor_size = executor_backlog.qsize()
            else:
                executor_size = "N/A"

            logger.info(
                "[IOLoop] pending_callbacks=%s timeouts=%s executor_queue=%s",
                pending_callbacks, timeouts, executor_size,
            )
            print(
                f"[FLOWER] IOLoop stats pending_callbacks={pending_callbacks} timeouts={timeouts} executor_queue={executor_size}")
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Failed to gather IOLoop stats: %s", exc)

        # Schedule next run 5 seconds later to keep monitoring alive even if
        # PeriodicCallback failed for some reason.
        try:
            self.io_loop.call_later(5.0, self._log_ioloop)
        except RuntimeError:
            # io_loop may be closing; ignore
            pass

    def start(self):
        self.events.start()

        # Patch Tornado HTTPServer to log when a TCP stream is accepted. This
        # will confirm whether connections are handed to Tornado at all.
        from tornado.httpserver import HTTPServer as _OrigHTTPServer

        if not hasattr(_OrigHTTPServer, "_flower_debug_patched"):
            orig_handle_stream = _OrigHTTPServer.handle_stream

            def _debug_handle_stream(self, stream, address, *args, **kwargs):  # type: ignore
                print(f"[FLOWER] HTTPServer accepted connection from {address}")
                return orig_handle_stream(self, stream, address, *args, **kwargs)

            _OrigHTTPServer.handle_stream = _debug_handle_stream  # type: ignore
            _OrigHTTPServer._flower_debug_patched = True  # type: ignore

        if not self.options.unix_socket:
            self.listen(
                self.options.port,
                address=self.options.address,
                ssl_options=self.ssl_options,
                xheaders=self.options.xheaders,
            )
            # Unconditional print so that we can see binding even when loggers are
            # filtered out by the environment.
            print(
                f"[FLOWER] Bound HTTP server on {(self.options.address or '0.0.0.0')}:{self.options.port}")
            logger.info("Flower listening on %s:%s", self.options.address or '0.0.0.0', self.options.port)
        else:
            from tornado.netutil import bind_unix_socket

            server = HTTPServer(self)
            socket = bind_unix_socket(self.options.unix_socket, mode=0o777)
            server.add_socket(socket)

        # Always start periodic observability callback. If you want to suppress
        # the output in production you can raise the logging level instead.
        PeriodicCallback(self._log_ioloop, 5000).start()

        self.started = True
        self.update_workers()

        # Schedule a very early callback (1 s) to confirm the IOLoop actually
        # runs. If we never see this line printed, the loop is blocked before
        # processing even the first timeout.
        self.io_loop.call_later(
            1.0,
            lambda: print("[FLOWER] IOLoop first tick executed – event loop is alive"),
        )

        print("[FLOWER] Calling io_loop.start() – server should now accept and process connections")
        self.io_loop.start()

    def stop(self):
        if self.started:
            self.events.stop()
            logging.debug("Stopping executors...")
            self.executor.shutdown(wait=False)
            logging.debug("Stopping event loop...")
            self.io_loop.stop()
            self.started = False

    def transport(self):
        """Return broker transport driver type without blocking the IOLoop.

        Instead of creating a new network connection (which may hang if the
        broker is unreachable) we parse the broker URL from the Celery config.
        If that fails we fall back to establishing a connection with a short
        timeout so that the operation is bounded.
        """
        broker_url = getattr(self.capp.conf, "broker_url", "")
        if "://" in broker_url:
            return broker_url.split("://", 1)[0]

        # Fallback – attempt to open a quick connection but *never* block for
        # too long. A 3-second timeout keeps the UI responsive even when the
        # broker is down.
        try:
            return getattr(
                self.capp.connection(connect_timeout=3.0).transport,
                "driver_type",
                None,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Unable to determine broker transport: %s", exc)
            return None

    @property
    def workers(self):
        return self.inspector.workers

    def update_workers(self, workername=None):
        return self.inspector.inspect(workername)
