# Failure Memory Review Demo

## Demo Title

```text
AI Coding Flight Recorder: review a patch with failure memory
```

## Promise

Given a patch or AI-generated change, `software-satellite-lab` recalls similar prior failures, shows evidence paths, asks for a human verdict, and records the outcome for future reuse.

## Demo Setup

Input artifacts:

- one git diff or patch summary
- one previous test failure event
- one previous repair event
- one accepted or rejected review note
- optional candidate output A/B

## Demo Flow

```text
1. satlab event ingest --patch changes.diff --note "Patch review input"
2. satlab recall failure --query "patch risk similar failure"
3. satlab pack run review-risk-pack --patch changes.diff
4. satlab verdict reject --event <id> --reason "Repeats prior missing-source bug"
5. satlab report latest --format md
6. satlab learning inspect --preview-only
```

The implementation covers the full local, file-first demo path. `pack run review-risk-pack` is intentionally a narrow built-in runner for this one declarative pack, not a general pack runtime. `satlab learning inspect --preview-only` writes inspection artifacts only; no trainable export is produced.

Optional proposal comparison:

```text
satlab compare proposals \
  --candidate proposal-a.md \
  --candidate proposal-b.md \
  --verdict winner \
  --winner-candidate 1 \
  --rationale "Proposal A preserves source evidence."
```

## Output Report Must Show

- patch summary
- top recalled similar failures
- source artifact paths
- prior verdicts
- backend / proposal comparison if present
- recommended next action
- human verdict
- learning-candidate state:
  - ready
  - needs_review
  - blocked
  - missing_source
  - excluded

## Demo Success Criteria

A viewer must be able to answer:

1. What happened?
2. Why is this risky?
3. Where is the source evidence?
4. What did the human decide?
5. Will this be reused later?
6. Why is it not automatically training data?

## Anti-Demo

Do not demo:

- agent writing code autonomously
- provider count
- fancy dashboard
- marketplace install
- live cloud orchestration
- automatic fine-tuning
