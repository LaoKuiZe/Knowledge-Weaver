# SPDX-License-Identifier: MIT

"""Reuse stateless HTTP connections without sharing Sessions across workers."""

import os
import threading

import requests

_LOCAL = threading.local()


def get_thread_session():
    """Return a process/thread-local transport with request-local credentials and no cookies."""
    if os.environ.get("SKILL_HTTP_CONNECTION_REUSE", "1") == "0":
        return requests
    pid = os.getpid()
    if getattr(_LOCAL, "pid", None) != pid:
        previous = getattr(_LOCAL, "session", None)
        if previous is not None:
            previous.close()
        _LOCAL.session = requests.Session()
        _LOCAL.pid = pid
    _LOCAL.session.cookies.clear()
    return _LOCAL.session
