# Durable Report Generation

The report feature is an additive execution path. Brief and Standard requests
continue through the existing synchronous `/ask` process. Report requests are
stored as durable jobs and consumed by a separate worker process.

## Standard report structure

The runtime `report-composer` skill and the `report-plan-v1` contract enforce:

1. Executive summary
2. Scope and evidence
3. Key findings
4. Zero or more analysis sections
5. Optional implications
6. Limitations
7. Conclusion

A standard report plan contains 5–8 sections and 900–1,400 words. The public
result contract permits 803–1,687 words so rounding across as many as eight
individually validated section tolerances cannot invalidate an otherwise valid
report. Assembly still enforces the tighter sum of the exact per-section bounds
for the current plan. Section word budgets are deterministically normalized to
the report budget before strict validation. Every section receives an explicit
evidence allow-list. The limitations section must cite typed limitation
evidence. Reports may contain up to three deterministic charts built from
tabular evidence; required charts must build successfully before assembly.
The final result revalidates the canonical order, chart-to-section bindings,
and exact citation coverage for both section and chart evidence.

## Generation phases

1. The existing evidence pipeline runs with `answer_mode=report`.
2. Verified evidence is frozen into a bounded, query-bound manifest.
3. One planning call creates the standard structure and evidence assignments.
4. Section calls run in parallel. Each section sees only its assigned evidence.
5. An invalid section receives at most one evidence-scoped repair call.
6. Code validates identities, word budgets, evidence references, and numeric
   grounding.
7. Code builds charts and assembles sections in plan order. There is no final
   LLM rewrite that could remove citations or alter validated claims.

The durable checkpoint contains the manifest, plan, and completed validated
sections. Retries skip the evidence pipeline, planner, and every already valid
section. Checkpoints are bound to the exact job query digest and capped at
1 MiB.

## Separate worker service

The existing Railway web service remains:

```text
python main.py
```

Run report processing as a separate service:

```text
python report_worker.py
```

Required worker settings:

```text
ENAI_REPORT_WORKER_ENABLED=true
ENAI_REPORT_WORKER_DB_URL=<write-capable PostgreSQL URL>
```

The worker database identity must be able to execute the versioned report-job
functions. Do not reuse the production API's dedicated read-only database
identity. The worker service also needs the normal evidence-pipeline database
and model-provider settings.

Optional bounded settings:

```text
ENAI_REPORT_WORKER_LEASE_SECONDS=900
ENAI_REPORT_WORKER_RETRY_DELAY_SECONDS=30
ENAI_REPORT_WORKER_POLL_INTERVAL_MS=2000
ENAI_REPORT_SECTION_MAX_WORKERS=4
```

The worker is disabled by default and is never started by `main.py`. Apply the
report-job database patch before enabling job creation or the worker.

## Rollout order

1. Apply the report-job schema/functions and execute the database contract test.
2. Deploy backend code with the worker disabled.
3. Deploy the frontend/Edge report endpoints with the Report selector hidden.
4. Start one worker service in staging and exercise completion, cancellation,
   lease expiry, retry, and resume.
5. Enable the Report selector for an internal cohort.
6. Review content-free plan evaluation, failure codes, latency, provider cost,
   and chart omission rates.
7. Expand the cohort only after the rollback and privacy checks pass.

Rollback is to hide/disable Report job creation, stop the separate worker, and
leave Brief/Standard processing untouched. Existing queued jobs remain durable
and can be resumed after re-enabling the compatible worker.
