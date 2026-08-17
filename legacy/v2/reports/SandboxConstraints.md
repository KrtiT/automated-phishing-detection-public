### Sandbox Constraints

The CLI environment cannot bind to localhost ports (`uvicorn` fails with `operation not permitted`). To keep validation realistic, the burst replay harness uses FastAPI's ASGI transport to exercise the same code path (router, model, scaler) without the OS binding step. In a deployable environment, the same `server.py` can be launched with `uvicorn --host 0.0.0.0 --port 8000` and fronted by a load balancer; the code does not rely on the ASGI shortcut outside the sandbox.

Nov 23 update: trying to run `uvicorn server:app --uds /tmp/uvicorn_phish.sock` also fails with `PermissionError: [Errno 1] Operation not permitted]` (log: `/tmp/uvicorn_phish.log`). The extended burst run (`burst_test.py --mode asgi --total 1023`) therefore still uses ASGI transport inside this CLI, but the same script can target a deployed uvicorn endpoint via `--mode http --base-url http://<host>:<port>` once ports are accessible.
