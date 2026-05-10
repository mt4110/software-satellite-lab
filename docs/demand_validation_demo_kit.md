# Demand Validation Demo Kit

This kit keeps public-demo validation local, file-first, and preview-only. It does not add an IDE extension, marketplace flow, cloud server, live provider integration, arbitrary pack runtime, vector search, or trainable export.

## What It Measures

- 10 source-linked dogfood events
- 5 failure-memory recalls
- 5 human verdicts
- useful-recall judgement rate
- critical false evidence count
- source path completeness
- human verdict capture friction
- clone-to-demo time
- external technical-user interview notes and counts

## Local Flow

```bash
python scripts/satlab.py validation template --output-dir artifacts/demand_validation_notes
python scripts/satlab.py event ingest --failure previous-failure.log --note "Prior failure memory"
python scripts/satlab.py recall failure --query "similar source-linked failure" --source-event-id <event-id>
python scripts/satlab.py verdict reject --event <event-id> --reason "Human review rejected this evidence."
python scripts/satlab.py validation record-run \
  --event <event-id> \
  --useful-recall yes \
  --critical-false-evidence-count 0 \
  --verdict-capture-seconds 20 \
  --notes-file artifacts/demand_validation_notes/dogfood_run_notes.md
python scripts/satlab.py validation record-interview \
  --participant user-1 \
  --recognized-pain yes \
  --wants-to-try yes \
  --notes-file artifacts/demand_validation_notes/external_user_interview.md
python scripts/satlab.py validation record-setup \
  --clone-to-demo-minutes 12 \
  --notes-file artifacts/demand_validation_notes/setup_timing.md
python scripts/satlab.py validation report --write --format md
```

## Gate Interpretation

`validation report` reads existing event, recall, verdict, dogfood-run, setup, and interview artifacts. The report is allowed to say `needs_data`; that is not a failure. It means the demo has code support, but demand has not yet been proven.

Keep moving only when the report shows:

- useful recalled evidence rate is at least 30%
- critical false evidence is 0
- source path completeness is at least 90%
- verdict capture stays under 30 seconds
- clone-to-demo stays under 15 minutes
- at least 3 external users recognize the exact problem
- at least 1 external user wants to try it on a repo
