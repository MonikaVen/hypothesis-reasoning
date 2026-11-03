# Beyond Retrieval

Experiment harness for the pre-results protocol *Beyond Retrieval: Evaluating Hypothesis-Driven Reasoning in Tool-Using Agents*.

The harness separates evidence acquisition, interpretation, and revision, and it freezes those boundaries before any confirmatory model run. It does not download AIT, OTRF, or OpTC, and it does not call a model. A synthetic outbound-transfer case exercises the runner. Those rows are a harness check, not a study result.

The design is in [docs/architecture.md](docs/architecture.md).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python -m beyond_retrieval plan
python -m beyond_retrieval run --backend local --experiment pilot --out runs/pilot
```

`plan` prints the protocol's run ceilings. `run` executes the synthetic case through the real runner.

`--backend local` is a deterministic tool user. It searches every visible source and answers from the rows it retrieved. It does not call a model, and its records are not a confirmatory result. Evidence-supported resolution stays `pending_adjudication` until a blinded review is attached.

`--experiment pilot` is V0 and V4 for conditions A and H. `scripts` adds AS and HS on V0. `synthetic` walks V0, V1, V2, V4, and V5 for A, H, AS, and HS.

A hosted model uses the OpenAI chat-completions shape:

```bash
export BR_API_KEY=...
export BR_MODEL=gpt-4o-mini          # optional
export BR_BASE_URL=https://api.openai.com/v1   # optional; any compatible host
python -m beyond_retrieval run --backend openai --experiment pilot --out runs/model-pilot
```

`BR_API_KEY` may also be named `OPENAI_API_KEY`. The key is not written into run records. `dry-run` still replays the scripted harness check.

## What is implemented

- Case manifests, environment splits, and lesson freeze
- Recorded-data variants V0, V1, V2, V4, and V5, with V3 refused
- One read-only query service shared by every condition
- Condition capabilities, action-contract checks, budget, and the audit graph
- Script execution with a validation gate that fails closed
- Mechanical scoring, blinded review packets, and the H1–H4 contrast definitions
- Cluster-aware effect estimates that do not claim population inference

## What is not implemented

Real dataset parsers, model clients, human adjudication UI, and Kyoushi regeneration. Those are extension points in the architecture doc. Filling a result table before a sealed run is intentionally unsupported: `unmeasured_main_table()` stays blank.

## License

Harness code is Apache-2.0. That license does not relicense any telemetry corpus. AIT Log Data Set V2.1 is CC BY-NC-SA 4.0; keep that term on the data if a later snapshot imports it. OTRF and OpTC keep their own release terms.
