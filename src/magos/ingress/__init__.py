"""Ingress: how requests enter magos. ``http`` is the FastAPI entry,
``mitm`` is the optional ``HTTPS_PROXY`` listener; both feed
:mod:`magos.routing`."""
