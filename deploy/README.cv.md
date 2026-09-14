# Compose face detection for M3-Bench

The existing `scripts/cv_service.py` remains an independent HTTP service.
Compose now builds and manages it: restart policy, health check, logs, and
an optional `with-cv` profile. It is not a dependency of ordinary memory runs.

## Start just CV

From the repository root:

```bash
docker compose -f deploy/docker-compose.cv.yml up -d --build --wait cv
curl --fail http://127.0.0.1:8081/health
export OGMEM_CV_SERVICE_URL=http://127.0.0.1:8081
# Run the M3 pipeline/benchmark from this same shell.
```

No database or LLM credentials are needed for this CV-only command.
Stop an existing manually launched CV service first if it occupies port 8081,
or choose a different host port:

```bash
OGMEM_CV_PORT=18081 docker compose -f deploy/docker-compose.cv.yml up -d --build --wait cv
export OGMEM_CV_SERVICE_URL=http://127.0.0.1:18081
```

`OGMEM_CV_PORT` changes the published port, not the container's internal 8081.
Keep using the chosen port for subsequent Compose commands, or put it in
`deploy/.env`. That file configures Compose; it does **not** export variables
to a benchmark process running on the host.

## Include CV in the existing stack

With the existing stack's `deploy/.env` and `ogmemory.yaml` configured:

```bash
docker compose -f deploy/docker-compose.yml --profile with-db --profile with-cv up -d --build --wait
```

Or start just the database and CV for a host-side M3 run:

```bash
docker compose -f deploy/docker-compose.yml --profile with-m3 --profile with-cv up -d --build --wait
```

The main Compose file still interpolates its required DB/LLM variables,
including services outside the selected profiles. Use the independent
`docker-compose.cv.yml` entry point when you only need CV and do not have
those variables configured. Both files reuse the same service definition
and project name; don't launch duplicate CV instances from them.

Host processes and the existing host-networked containers reach CV through
`http://127.0.0.1:8081` (or the chosen host port). Another bridge-networked
container must join CV's Compose default network and use `http://cv:8081`;
its own `127.0.0.1` does not refer to the host.

## Logs, stop, and request smoke test

```bash
docker compose -f deploy/docker-compose.cv.yml logs --tail=100 cv
docker compose -f deploy/docker-compose.cv.yml ps cv
curl --fail -H 'Content-Type: application/octet-stream' \
  --data-binary @/path/to/test-frame.jpg \
  'http://127.0.0.1:8081/detect/faces?threshold=0.5&max_items=20'
docker compose -f deploy/docker-compose.cv.yml stop cv
```

Replace the image path before running the request. An empty `detections`
array can be a valid response; it does not establish detection accuracy.
Use service-scoped `stop cv` rather than bringing down the whole shared
project. Rebuild and repeat the `up` command after changing the service.

## Deliberate limits

- The supplied image explicitly runs **CPU OpenCV Haar**, matching the
  script's default. It does not install InsightFace, return face embeddings,
  download model weights, or use an NPU. If your previous service used
  InsightFace, switching to this image changes the detector and is **not**
  a like-for-like benchmark comparison.
- Fix the detector and all other extraction settings across accuracy
  experiments. Containerizing the service alone is not an accuracy gain.
- `/health` checks that the selected backend initialized. Test actual
  `/detect/faces` requests before committing to a full benchmark.
- Object and pose endpoints in the existing script are still stubs.
- There is no authentication in this service. The host port binds to
  loopback by default; do not expose it publicly. Remote access needs a
  separately secured network/proxy, not just a `0.0.0.0` bind.
- The image runs as a non-root user, drops Linux capabilities, and copies
  only the CV script and dependency file—not videos, database credentials,
  or the host Python environment. Dependency versions are pinned at the
  direct-dependency level; this is not a fully locked dependency tree.
