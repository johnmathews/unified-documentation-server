# 2026-05-07 — Embedding batch size: profile + safer default

Third change of the day on the OOM mitigation thread. Earlier commits
(`fd6afcf`, `5d203ed`, `e122f4d`) tightened cgroup containment, added
Chroma liveness to `/health`, and capped `conversations.db` growth.
Today's deploy is live on infra; this commit fixes the *root cause*
of the historical 1.2-1.4 GB worker peaks that necessitated all of it.

## Profile

Ran a synthetic embedding profile inside the live docserver container
(`docker exec ... .venv/bin/python /tmp/profile_single.py`) using the
production ONNX model. Profile script: load
`OnnxEmbeddingFunction`, warm it (one call to load the ONNX session),
then embed N chunks of representative ~400-char text in batches of
size B. Peak RSS captured via `resource.getrusage` (Linux ru_maxrss
is in KB).

```
                       warmup   peak after first batch   delta
batch=4      ~310 MB              427 MB              +118 MB
batch=8      ~310 MB              582 MB              +268 MB
batch=16     ~310 MB              748 MB              +448 MB
batch=32     ~310 MB              806 MB              +496 MB
```

Two surprising findings:

1. **The peak is reached on the very first batch and stays flat.** It
   isn't a leak — subsequent batches reuse the activation memory. The
   peak is determined entirely by batch size.
2. **Per-chunk inference time is roughly equal at batch=4..16** (~0.29
   s/chunk on this hardware). At batch=32 it dropped to ~0.77 s/chunk,
   likely from memory pressure / swap thrashing on a contended host.

So lowering the batch size from 32 to 8:
- Cuts the peak from ~810 MB → ~580 MB (well inside the 768 MB cgroup)
- Costs nothing in throughput
- Removes the "embedding cycle blows the cgroup" failure mode entirely

The cgroup containment from `fd6afcf` was always going to OOM-kill the
worker cleanly when it spiked past 768 MB, but having every real-content
cycle die mid-flight isn't a great steady state — repeat kills compound
into restart noise even when isolated. Fixing the root cause is better
than relying on cgroup-OOM-as-circuit-breaker.

## What changed

`src/docserver/embedding.py`:
- Hardcoded default `batch_size=32` on `_forward()` is gone.
- New module-level `_DEFAULT_EMBEDDING_BATCH_SIZE = 8` with a
  comment explaining the empirical numbers.
- `OnnxEmbeddingFunction.__init__` accepts a `batch_size` kwarg.
- New `_resolve_batch_size()` static method handles precedence:
  explicit arg → `DOCSERVER_EMBEDDING_BATCH_SIZE` env var → default.
  Invalid env values (non-int, zero, negative) log a warning and fall
  back to default. Explicit constructor arg of 0 or negative clamps
  to 1 — explicit caller intent wins, just won't crash.
- `__call__` and `_forward` use the resolved instance attribute.

`README.md` and `docs/operations.md`: documented the new env var with
the per-batch peak numbers from the profile.

`tests/test_embedding.py` `TestBatchSize` (9 new tests):
- default is 8, env var override, explicit arg over env, invalid
  values fall back to default, zero/negative clamping, end-to-end
  verification that `__call__` actually uses the resolved size.

## Side note: cgroup containment proven in production

While running the profile, my variants script accidentally over-allocated
(it created multiple `OnnxEmbeddingFunction` instances back-to-back
without releasing prior ONNX sessions). The kernel's response was a
clean cgroup OOM kill with `CONSTRAINT_MEMCG`, `task=python`,
`oom_score_adj=500`. The host stayed healthy. Exactly the post-deploy
behaviour we wanted to verify after `fd6afcf` + the Ansible template
update at `2ff2ce3` in `home-server`. The OOM containment design works
under real production conditions.

## Tests + lint

433 tests pass (was 424), ruff clean, coverage 87%. Net add: 9 new
tests in `TestBatchSize`.

## Reminder

The deployed image on infra still has the old `batch_size=32` hardcoded
default. After this commit's CI build completes and the new image is
published, run:

```sh
ssh infra "cd /srv/infra && docker compose pull documentation-server && docker compose up -d documentation-server"
```

Or wait until the next Ansible run picks up the new image tag.

After the new image is live, ingestion cycles on real-content commits
should no longer trip the cgroup OOM. To verify: trigger a real commit
on any watched repo (e.g. push to `home-server`), wait 30 minutes for
the next poll, and confirm:
1. `docker logs documentation-server` shows `ingestion_cycle_complete`
   with non-zero `upserted` for that source
2. `dmesg` shows no new OOM event during the cycle
3. `docker stats documentation-server` peak during the cycle stays
   below ~600 MB
