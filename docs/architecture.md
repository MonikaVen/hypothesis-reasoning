# Experiment architecture

This package is the control plane for the Beyond Retrieval protocol. A later study plugs in dataset parsers and model clients. The boundaries below are already enforced, because the scientific claims depend on them.

The protocol's question is not whether an agent can finish an investigation. It is whether a gain survives equal tools, equal facts, and equal budget, and whether the agent changes its answer when the evidence changes. The code is arranged so those comparisons are properties of a run spec, not of a prompt.

## Layers

Dependencies point inward. Policies cannot open a database. Scoring cannot see a condition name unless a caller deliberately passes one. The grid cannot execute a run.

```text
cli
  -> runner          one spec, one mount, one policy
  -> grid            allocations only
  -> fixtures        synthetic case, not a result

runner
  -> evidence        snapshots and variants
  -> tools           query service and script sandbox
  -> agent           budget, contracts, audit graph, guidance
  -> scoring         mechanical checks and blinded packets
  -> corpus          splits, playbooks, lesson freeze

grid, scoring, analysis, evidence, tools, agent, corpus
  -> models, hashing, errors
```

`analysis` does not import the runner. It consumes trial values that a caller has already decided are comparable. `memory` is the optional representation experiment and is not on the investigation path.

## Two objects for one case

`CaseBundle` is the curator object: events, distractors, rubrics, rename map, and the V4 cut. `TaskView` is the only thing a policy receives: question, alert, tool names, and a guidance string.

The runner is the only component that holds both. `execute_run` never puts the rubric, the cut, or the `unrelated` flag on the view. Agent-facing rows are `AgentRecord`s. Extra fields such as `attack_label` are rejected by the model, and curator leak markers are rejected when the mount is built.

A graph edge is an assertion. `GraphEdge` has `asserted_by` and `script_validated`. It has no field that means the relation is true. Scoring does not read the graph as ground truth.

## One run

```text
RunSpec + sealed? protocol
        |
        v
capabilities for (condition, ablation, mode)
        |
        v
guidance string   <-- reference, playbook, or frozen lessons
        |              H cannot receive lessons; HS must
        v
mount for the variant and, for V5, the stage
        |
        v
loop: propose -> charge tokens -> authorize -> contract check
      -> query or script -> charge the call -> audit graph
        |
        v
FinalAnswer
        |
        v
mechanical score on visible, retrieved record ids
        |
        v
ESR = pending until a blinded adjudication is attached
```

`RunSpec` names the experiment, case, variant, condition, backbone, repeat, ablation, and mode. The condition on the spec wins over anything the policy object claims. A policy that asks for a script under condition A is denied, and the denial consumes budget.

### Budgets and failures

Default caps are 40 tool calls, 30,000 generated tokens, and 20 minutes. Each script call is limited to 120 seconds. V5 splits the cumulative caps in half across two stages and keeps the per-call script limit.

Invalid calls, contract repairs, and validation reruns all consume calls. Exhausted budget and agent exceptions stay in the scored set (`counts_as_missing=false`). An infrastructure outage is `missing`, with an exclusion reason that says to rerun the same spec. The runner does not drop a run because the answer looks bad.

### Action contracts

Conditions H and HS must attach a contract to every query and script. The checker is structural: a specific question, a tool, a discrimination statement, and either two hypotheses with different predicted observations or one named hypothesis plus an open-set alternative. "The tool will provide useful information" is rejected. A rejected contract consumes a call and does not execute. The checker does not decide whether the prediction is true.

A falsification flag is recorded and is not an ESR credit. Two runs with the same answer and different flags get the same mechanical score.

## Conditions

Ablations remove a capability the condition already has. They do not grant scripts, lessons, or contracts.

| ID | Queries | Scripts | Validation | Graph | Contracts | Lessons | Guidance |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R | fixed plan | no |  |  |  |  | fixed retrieval |
| A | yes | no |  |  |  |  | domain facts |
| P | yes | no |  |  |  |  | same facts, as a playbook |
| H | yes | no |  | yes | yes | no | domain facts |
| AS | yes | yes | no |  |  |  | domain facts |
| HS | yes | yes | yes | yes | yes | yes | facts plus frozen lessons |

H versus A is the structure contrast (H1). HS versus AS holds script access constant (H3). HS versus HS with `no_script_validation` is H4. H2 is synthesis mode: the same bundle, no tools, standard (A) versus structured (H).

Playbook steps of kind `fact` must cite proposition ids from the shared reference. A reference proposition that contains an incident-specific token is rejected. Lessons must cite training cases only, and the frozen set is content-hashed.

Synthesis mode clears queries, scripts, and contracts. The policy's view has an empty tool list, and the run records no steps. No-evidence mode keeps the condition's query tools and mounts zero incident rows.

## Evidence

Variants are pure functions of a bundle and a seed. They are built before the policy runs, then hashed.

| Variant | What the code does | What it refuses |
| --- | --- | --- |
| V0 | The base events |  |
| V1 | Bijective rename of hosts, users, and IPs. Command strings in `preserve_exact` stay literal. The rubric text is renamed with the same map. | A mapping that is not one-to-one |
| V2 | Adds unrelated rows from the same environment, up to a 2× target | Rows from another environment, or rows that are not marked unrelated |
| V3 | Not a transform | `CounterfactualRequiresRegeneration` |
| V4 | Deletes a cut | A cut that leaves any modeled support path for the blocked finding |
| V5 | Stage 1 is the incomplete bundle. Stage 2 materializes a new database that also contains the release. | Overlapping stages, or a script file that still contains stage-2 ids during stage 1 |

Support paths live on the V0 rubric (`evidence_paths`). V4 is valid only when every path for the blocked finding intersects the cut and none survives. Hidden ids and unknown ids return the same `NOT_FOUND` string. The query service never says that a row is withheld.

The query tools are `list_sources`, `describe_schema`, `search_events`, `get_records`, and `aggregate_events`. Ordering is `(timestamp, record_id)`. Pages default to 200 rows and 12,000 characters. `group_by` is an allow-list, not a SQL fragment. Parser failures are a `parser_status` value and are not dropped.

Script mounts are sqlite files built from the visible rows only. The sandbox is a local `python -I` process with a timeout. It is a process boundary for analytical scripts, not a multi-tenant isolation product. HS validation runs the script against a known-answer fixture the validator owns, then an independent check on the case database. If no validator is configured, HS fails closed and admits nothing. AS records `validation=not_applicable` and does not admit a verified artifact. Grading still uses the rubric, not the script's own test.

## Experiments

Ceilings, from `grid.py`:

| Experiment | Comparison | Ceiling |
| --- | --- | --- |
| E1 | A, H, AS, HS on recorded variants | 80 × 5 × 4 × 2 × 3 = 9,600 |
| E2 | Standard vs structured synthesis | 80 × 2 × 2 × 3 = 960 |
| E3 | HS without validation; reuse matched HS from E1 | 240 new runs |
| E4 | The same four conditions on OTRF and OpTC V0 | 1,152 |
| Planned | E1–E4 | 11,952 |

The actual E1 count is `4 × backbones × repeats × eligible case-versions`. At the planned widths that factor is 24. Ineligible variants are omitted, not scored as failures. V3 is not an eligibility entry.

Outside that ceiling, and rejected if someone adds them to E1–E4: the R/P subset (240), the no-evidence diagnostic (480), pilot 1 (80), pilot 2 (40), exploratory techniques (hypothesis tree, critique, replanning, and the rest of that list), and the optional regenerated cohort (576).

Confirmatory execution requires `ProtocolFreeze.sealed`. The freeze hashes guidance, lessons, the tool schema, rubrics, caps, and `analysis_spec_hash()` over H1–H4. Changing a contrast changes the hash.

## Scoring and analysis

Mechanical checks cover an admissible verdict, every mandatory finding id, citations that exist in the visible mount, a non-empty answer, and probabilities over `malicious`, `benign`, and `insufficient_evidence`. Citations the agent never retrieved are flagged and do not, by themselves, fail the check. Entailment and material invention are `needs_human`.

ESR is `pending_adjudication` until `apply_adjudication`. A pass requires the mechanical check, `support_holds`, and no material unsupported claim. `blind_packet` drops condition, ablation, and contracts. Reviewers see the answer and the cited rows.

`revision_label` is a mechanical tag for V5: correction, resolution, stable pass, or harmful reversal. It is not confirmatory until both stages are adjudicated.

`analysis.effect_estimate` averages repeats inside a case, weights cases equally, and also reports an environment macro-average. Bootstrap resamples campaigns. `population_inference` is false. `conventional_significance_attainable` is false at four or fewer environments, because environment-level label swaps then have at most 16 assignments. `holm` adjusts a caller-supplied p-value vector. This module does not invent p-values. `unmeasured_main_table()` is the reporting template; every outcome cell is an em dash.

## Extension points

**A real corpus.** Implement a loader that returns `CaseBundle`s and a list of `CaseRecord`s. Call `audit_environment_splits` before any lesson is frozen. Keep train environments (`fox`, `harrison`), development (`russellmitchell`, `santos`), and test (`shaw`, `wardbeck`, `wheeler`, `wilson`) in `AIT_ENVIRONMENTS`. Do not put test answers in the loader's agent projection. Normalization must keep `parser_status` rather than drop a bad line.

**A backbone.** Implement `Policy`: `start(TaskView)`, `propose(ToolResult | None) -> Proposal`, `finalize() -> FinalAnswer`, and `notify_release()` for V5. Read tools only through the proposals the runner executes. Put provider token counts on `Proposal.generated_tokens` and record `token_accounting="provider"` when the count is real. The scripted policies in `scripted.py` are the test double, not condition A or H.

**A source-specific parser.** Teach it to emit `Event`s. OpTC duplicate process objects and path forms stay in that adapter. Do not merge processes because the executable name matches, and do not repair a missing edge from the rubric.

**Regeneration.** A new module may emit paired bundles for V3. It must not call `prepare_mount(..., V3)`. The recorded-data runner will keep refusing that variant.

**Representation experiment.** `memory.render_representation` renders one fact list as a ledger, a relation list, or a typed graph. Every rendering emits the same fact ids. `corrupt_edges` swaps or deletes 0, 10, or 20 percent of derived edges and leaves them marked derived. It is not on the E1 path.

## Synthetic harness

`fixtures.outbound_case` is the appendix's ambiguous outbound transfer, with made-up rows: a flow, a translation, a `curl` process, and a backup job on a different host and window. V4 removes the translation and the process, which are the two modeled paths to the internal host. The dry run walks conditions A and H on V0 and V4 with a frozen clock and no model. Use it to see a trace. Do not cite it as evidence about agents.
