# Legacy dashboard

This folder holds the original Python-served dashboard (`dashboard_server.py` +
`dashboard/`). It has been superseded by the Next.js app in `app/` and
`components/`, which is what the live demo at
https://neuralforge-vercel.vercel.app runs.

It is kept for reference and still works locally:

```bash
python legacy/dashboard_server.py --port 8080
# then open http://localhost:8080
```

It reuses `sw/fpga_simulator.py` and `weights/` from the repo root, so run it
from a full checkout.
