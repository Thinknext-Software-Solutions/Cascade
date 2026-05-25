"""Cascade Studio: a web dashboard for the Cascade pipeline.

This sub-package bundles the FastAPI backend and (pre-built) Next.js
frontend that ship together as part of cascade-agent. Install with the
[studio] optional extra:

    pip install cascade-agent[studio]

Then run:

    cascade ui

A local web server starts at http://localhost:8000 and your browser opens
to the dashboard.

Development workflow:
    The frontend source lives in a separate repo (cascade-studio). To
    update what gets shipped:
      1. cd cascade-studio && npm run build
      2. cp -r out/* /path/to/cascade-agent/src/cascade/studio/static/
      3. Release a new cascade-agent version
"""
