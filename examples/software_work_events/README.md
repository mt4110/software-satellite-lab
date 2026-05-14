# Software Work Event Example Gallery

These examples show small public `software_work_event` records that can be inspected without private design notes, API keys, network calls, or generated training artifacts.

Each JSON file points at a local public source file under `examples/software_work_events/sources/`. The examples are intentionally compact so contributors can see how event status, target paths, content notes, tags, and source references fit together.

## Examples

| File | Event kind | Purpose |
|---|---|---|
| `patch_input_needs_review.json` | `patch_input` | A local patch has been captured and still needs review. |
| `prior_failure_risk.json` | `test_failure` | A source-linked negative prior can support risk review. |
| `verification_pass.json` | `verification` | A local test log records positive verification evidence. |
| `human_verdict_reject.json` | `human_verdict_recorded` | A human rejection is captured as recallable evidence. |

## Local Check

The unit test suite validates these examples with `build_event_contract_check`. To run only that focused check:

```bash
PYTHONPATH=scripts python3 -m unittest tests.test_software_work_events.SoftwareWorkEventTests.test_public_gallery_examples_satisfy_event_contract
```
