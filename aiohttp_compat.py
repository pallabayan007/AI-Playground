"""Compatibility shim for `aiohttp` symbols expected by some vendor code.

Place this module in the project root and import it at the top of scripts
before importing `openai` or other libraries that may transitively import
`aiohttp`. That ensures symbols like `SocketTimeoutError` and
`client_exceptions.NonHttpUrlClientError` exist across aiohttp versions.
"""
def _apply():
    try:
        import aiohttp
    except Exception:
        return

    # Ensure aiohttp has SocketTimeoutError (older/newer name differences)
    try:
        if not hasattr(aiohttp, "SocketTimeoutError"):
            setattr(aiohttp, "SocketTimeoutError", getattr(aiohttp, "ServerTimeoutError", TimeoutError))
    except Exception:
        pass

    # Ensure client_exceptions module exists and has expected attributes
    try:
        ce = getattr(aiohttp, "client_exceptions", None)
        if ce is None:
            try:
                from aiohttp import client_exceptions as _ce
                ce = _ce
                aiohttp.client_exceptions = ce
            except Exception:
                class _Dummy:
                    pass
                ce = _Dummy()
                aiohttp.client_exceptions = ce

        if not hasattr(ce, "NonHttpUrlClientError"):
            setattr(ce, "NonHttpUrlClientError", getattr(ce, "InvalidURL", Exception))

        if not hasattr(ce, "InvalidUrlClientError"):
            setattr(ce, "InvalidUrlClientError", getattr(ce, "NonHttpUrlClientError", getattr(ce, "InvalidURL", Exception)))
    except Exception:
        pass


# Apply automatically on import
_apply()
