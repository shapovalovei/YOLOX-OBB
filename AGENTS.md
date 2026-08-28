# Agent Instructions for YOLOX-OBB

## Mission and authority

This repository is a maintained fork of YOLOX-OBB. Oriented bounding boxes
(OBB) are a core contract; treating this project as a generic horizontal
bounding-box (HBB) detector is incorrect.

Mobile and bank-card deployment are important practical directions, but this
remains a general maintained OBB codebase. Do not add card-specific assumptions
unless the task explicitly establishes them.

The human project owner or orchestrator controls scope, acceptance criteria,
semantic completion, merge, and Issue closure. Follow explicit owner
instructions and report resulting risks when they override this file.

## Default sequence

For meaningful work, follow:

```text
task → research current state → read relevant Issues, PRs, and history
→ understand the contract → define the problem or hypothesis
→ implement only within authorized scope → validate → report evidence
```

Research before implementation is the default. Do not jump from a task
description directly to editing code.

## Start-of-task repository gate

Before editing, inspect:

- `git status --short --branch`;
- the current branch, HEAD, and configured remotes;
- relevant tracked and untracked changes;
- implementation, call sites, tests, and configuration in the task area.

Distinguish pre-existing human changes from your own. Preserve unrelated work
and report it; do not silently overwrite, clean, or reformat it.

Never use destructive operations against user work, including `git reset --hard`,
destructive `checkout` or `restore`, `git clean`, force-push, or rewriting
published history.

## Research gate

Research must be proportional but sufficient to avoid reverting or duplicating
maintained behavior. For meaningful changes, inspect as relevant:

- current implementation, important call sites, and regression/contract tests;
- open Issues, related CLOSED Issues, bodies, comments, links, and acceptance
  criteria;
- relevant Pull Requests in all available states;
- `git log`, `git show`, `git blame`, and previous fixes for the affected
  invariant.

Use repository search and any authenticated or read-only GitHub mechanism; the
`gh` CLI is optional. If Issue or PR state cannot actually be accessed, say so
explicitly and do not claim that it was inspected.

Before implementation, be able to state the current behavior, evidence for the
defect or request, the unchanged contract, files/subsystems in scope, and the
acceptance criteria and validation that will prove completion.

Do not research the entire repository mechanically for a tiny, unambiguous
change. Investigate broadly when the invariant, history, or compatibility
behavior is unclear.

## Operating modes

### Investigation mode

Use this mode when the cause is unclear, multiple designs are plausible,
architecture or export behavior is uncertain, OBB conventions or training
semantics may change, compatibility is unclear, or history may explain unusual
code.

```text
investigate → report findings and options → obtain approval → implement
```

Do not implement a speculative solution while a material design decision is
unresolved.

### Bounded implementation mode

Proceed autonomously only when scope is explicit, expected behavior and
acceptance criteria are established, no architectural or OBB contract decision
is unresolved, and the change can remain bounded.

For correctness or bug work, prefer:

```text
reproduce or failing regression → minimal causal fix → passing regression
```

Do not require a new failing test for every change. For features, export work,
or refactors, use the existing contract and Issue criteria, then add focused
tests proving requested behavior and preserved compatibility.

## OBB contract and invariants

Protect OBB semantics across `yolox/models`, `yolox/data`, `yolox/utils`, tests,
tools, export, and deployment paths. Contract-sensitive behavior includes:

- center coordinates, width/height semantics, and polygon geometry;
- angle semantics with explicit degree/radian conversion;
- label and prediction/output field ordering;
- rotated assignment and candidate geometry;
- augmentation, Mosaic, MixUp, and flip angle preservation;
- grid/stride decode and KLD prediction/target semantics;
- rotated postprocess and NMS.

Valid tensors can still represent a regression if geometry conventions change.
Do not reinterpret OBB data as HBB data or silently swap width/height, angle
units, angle normalization, field ordering, or polygon point meaning.

Deliberate changes to an OBB convention, public output layout, or compatibility
contract require owner/orchestrator approval and dedicated compatibility or
regression evidence.

Rotated NMS and geometry postprocess may remain outside the exported neural
network graph when that improves backend portability. Do not force rotated
operations into exported graphs merely for architectural neatness.

## Donor and reference repositories

`DDGRCF/YOLOX_OBB` is a donor/reference implementation only. Use it for
alternative ideas, OBB clues, or deployment hypotheses when useful.

Use this reasoning chain:

```text
donor observation → local hypothesis → inspect maintained code/history/tests
→ prove local need → bounded local solution
```

Do not infer that donor behavior belongs here because it is newer or different.
Do not broadly merge, blindly cherry-pick, or copy donor architecture. The
maintained fork's code, tests, Issues, and explicit requirements take
precedence.

## Validation discipline

Match validation to the change:

- Bug/correctness fix: reproduce where feasible, add or update focused
  regression coverage, run it, then run broader maintained coverage.
- OBB geometry or training-path change: run relevant OBB regressions and the
  full maintained suite where the environment permits.
- Export/deployment change: preserve reference behavior; test runtime behavior,
  output layout, OBB geometry, and static compatibility when changing dynamic
  behavior.
- Documentation-only change: validate formatting, relevant links/commands,
  diff, and repository hygiene; do not imply model validation occurred.

Inspect actual test and CI commands before treating them as authoritative.
Report every skipped, blocked, or partial check with its reason. Never hide real
failures as warnings.

Deterministic tests can prove a code defect, geometry violation, or
export/runtime mismatch. They do not by themselves prove a trained model is
better. Separate code correctness from downstream model-quality evidence.

## Training, datasets, and artifacts

Without explicit authorization, do not:

- start training, QAT, or expensive GPU work;
- modify dataset membership, generate datasets, or download large datasets;
- download large checkpoints or model assets;
- create large sweeps or unrelated experiments;
- commit checkpoints, ONNX models, TensorRT engines, generated datasets, or
  other generated model artifacts.

PTQ and other bounded deployment experiments require explicit task or Issue
authorization and must stay within scope. Training-quality experiments should
normally live in the separate training project with their own provenance and
controlled evaluation.

## Scope and approval gates

Do not combine unrelated cleanup, refactoring, or compatibility work with a
requested task. Report separate defects and recommend a separate Issue instead
of silently fixing them.

Stop and ask the owner/orchestrator when:

- scope or acceptance criteria expand or conflict with repository facts;
- multiple incompatible designs remain plausible;
- an OBB convention, public API, output layout, or compatibility contract would
  change;
- dataset changes, training, QAT, expensive compute, large downloads, or a
  major dependency are needed;
- a broad refactor or removal of historical compatibility is proposed;
- an important invariant cannot be proven;
- donor behavior conflicts with maintained tests/history;
- unrelated local changes overlap the implementation area;
- destructive Git operations or published-history rewriting would be needed.

Do not interrupt for ordinary read-only research, local reproduction, focused
tests, or a minimal implementation already authorized by a precise task.

## Issues, branches, commits, and Pull Requests

Issues and PRs are engineering context, not optional administrative metadata.
Use Issues for changing scope, decisions, acceptance criteria, and durable task
state. Use PRs for implementation rationale, diff review, and validation
evidence. Do not hard-code current Issue or PR numbers here.

Before creating an Issue, search OPEN and CLOSED Issues and relevant PRs for
duplicate, overlapping, superseded, or already-tracked work. If a canonical
Issue exists, use or cross-link it. Do not create parallel Issues for the same
engineering question merely because wording, evidence, or context differs.

Codex may add factual engineering evidence—progress, investigation evidence,
validation results, and commit/PR links—to an existing Issue or PR when the
current task explicitly authorizes GitHub workflow. Scope, decisions, and
completion remain owner/orchestrator authority: approval is required before
creating a new Issue unless explicitly authorized, changing scope or criteria,
making project/product decisions, declaring completion, closing an Issue,
merging a PR, or marking unresolved work complete.

For meaningful implementation work, use:

```text
Issue → research → purpose-specific branch → implementation → validation
→ logical commit(s) → PR → review → Issue evidence → owner merge and closure
```

Direct commits to `main` are prohibited by default. Prefer short-lived names
such as `fix/<issue>-<slug>`, `feat/<issue>-<slug>`, `test/<issue>-<slug>`, or
`docs/<issue>-<slug>`, without making naming needlessly rigid.

Commits must be logically scoped, reviewable, imperative, and free of unrelated
formatting or generated artifacts. Use `Refs #N` for ongoing or partial work.
Use `Closes #N` only when closure is authorized and all acceptance criteria are
actually satisfied.

Codex may open a Draft PR only when the task or orchestrator explicitly
authorizes that external action. Codex must not merge its own PR, mark
unresolved work ready or complete, close an Issue, or rewrite published
history. Those semantic actions belong to the owner or an authorized maintainer.

## Completion report

Report evidence, not only a conclusion. Include as applicable:

- initial and final worktree state;
- branch, commit, Issue, and PR;
- research sources, relevant history, and hypothesis or requested behavior;
- actual result, changed files, and intentionally unchanged scope;
- exact validation commands and pass/fail/skipped results;
- artifacts created outside Git, limitations, risks, and decisions still needed.

Distinguish FACT, INFERENCE, and UNVERIFIED ASSUMPTION whenever uncertainty
could affect the engineering decision.
