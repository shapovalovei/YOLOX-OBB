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

## Project ownership boundaries

This repository is one of three related but separate projects. Evidence may
cross these boundaries, but implementation belongs in the repository that owns
the responsibility. Do not infer implementation scope from a related Issue,
downstream dependency, or convenient checkout location.

### YOLOX-OBB: maintained framework and generic OBB behavior

Repository: <https://github.com/shapovalovei/YOLOX-OBB>

`YOLOX-OBB` owns improvement of the maintained model framework itself. Work
that belongs here includes:

- assignment logic and rotated candidate geometry;
- OBB geometry correctness, losses, and angle semantics;
- augmentation, Mosaic, and MixUp behavior;
- model and head implementation;
- generic inference, decode, postprocess, and rotated NMS correctness;
- framework-level export correctness and general dynamic/static export behavior;
- regression tests for framework behavior, compatibility fixes, and maintained
  framework engineering.

This repository is not the place for training a specific production model,
choosing a production checkpoint, A/B training or model-quality experiments,
calibrating a trained artifact, evaluating a specific FP32/FP16/INT8 candidate,
or implementing React Native, Android, or iOS SDK behavior.

Useful rule: if the question is whether YOLOX-OBB itself behaves correctly or
should be improved generically, it belongs here.

### card-detector-training: concrete trained models and provenance

Repository: <https://github.com/shapovalovei/card-detector-training>

`card-detector-training` owns the concrete trained model and its experimental
and deployment provenance. Work that belongs there includes:

- training, resuming training, recipes, and controlled A/B experiments;
- dataset membership for a concrete experiment;
- checkpoints, checkpoint selection, model-quality evaluation, and real
  validation evidence;
- export of a particular trained model, including FP32, FP16, TFLite/LiteRT,
  and INT8/PTQ conversion or evaluation;
- QAT when separately authorized, calibration, artifact hashes, provenance,
  and deployment-quality comparison of concrete trained artifacts.

`YOLOX-OBB` may own generic export correctness; `card-detector-training` owns
exporting and evaluating a particular trained model. If the question is how a
specific model performs, is trained, selected, converted, quantized, or
packaged as a model artifact, it belongs there.

### react-native-scanner-sdk: final SDK and product integration

Repository: <https://github.com/shapovalovei/react-native-scanner-sdk>

`react-native-scanner-sdk` owns the final SDK and product integration layer.
Work that belongs there includes:

- React Native, Android, and iOS implementation;
- model asset packaging inside the SDK;
- runtime integration, SDK-owned native preprocessing, detector invocation,
  and SDK-specific OBB decode/NMS when required;
- source-frame mapping, camera/ROI integration, lifecycle, and threading;
- SDK API behavior, physical-device qualification, and release/package
  integration.

If the question is about consuming a model inside the actual SDK or device
application path, it belongs there. This repository may be inspected as
downstream reference when a framework task needs to understand an external
contract, but that does not turn the task into SDK implementation.

Cross-project evidence may inform a task. Cross-project implementation
requires an explicit task in the repository that owns that work. Use this
sequence:

```text
external observation → local hypothesis → determine owning repository
→ inspect local implementation/history/tests → implement only in that repository
```

Do not modify another repository during a task unless the orchestrator
explicitly authorizes cross-repository work. In particular, do not
autonomously follow a chain such as `framework → training → export → SDK`.

## Project-boundary gate

Before doing meaningful work, answer these questions internally:

1. Is this a generic YOLOX-OBB framework or model-correctness problem?
2. Is this a concrete trained-model, training, export, or quality problem?
3. Is this an SDK or product-integration problem?

Select the owning repository before implementation. If the current repository
does not own the requested implementation, do not implement it here. Report
the ownership mismatch, identify the correct repository, and request
orchestrator approval before moving the work. The existence of a downstream
related Issue is not authorization to switch repositories.

## Default sequence

For meaningful work, follow:

```text
request → project-boundary gate → research current state
→ search OPEN/CLOSED Issues and relevant PRs and history
→ identify or create the canonical Issue → confirm scope and acceptance criteria
→ understand the contract → define the problem or hypothesis
→ implement only within authorized scope → validate → commit → PR
→ post factual Issue evidence → orchestrator review
→ owner merge / semantic completion / Issue closure
```

Research, ownership, and the canonical Issue gate always precede
implementation. Do not jump from a task description directly to editing code.
Bounded implementation mode may proceed autonomously only after these gates
are satisfied.

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

Before implementation, be able to state the owning repository, current
behavior, evidence for the defect or request, the unchanged contract,
files/subsystems in scope, the canonical Issue, and the acceptance criteria and
validation that will prove completion.

The lifecycle is explicitly:

```text
RESEARCH → establish ownership → establish current behavior/history
→ establish canonical Issue/scope → formulate hypothesis or requested behavior
→ IMPLEMENT → validate
```

Research is always first. For unclear tasks, report findings and obtain the
orchestrator decision before implementation. For clear tasks, bounded
implementation begins only after research and the Issue gate; “bounded” does
not bypass them.

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

Proceed autonomously only after the project-boundary, research, and
canonical-Issue gates are satisfied, scope is explicit, expected behavior and
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

### Mandatory Issue-first workflow

Every meaningful engineering task that will result in repository changes must
have a canonical Issue before implementation begins. This applies to code,
tests, export behavior, CI/workflow, engineering documentation, and repository
governance. Tiny read-only questions or pure investigation that produces no
repository changes do not require a new Issue merely to inspect the project.

Follow this order:

```text
request → research current state
→ search OPEN/CLOSED Issues and relevant PRs
→ identify the existing canonical Issue or create one
→ confirm scope and acceptance criteria → implementation
→ validation → commit → PR → factual Issue evidence/comment
→ orchestrator review → owner merge / semantic completion / Issue closure
```

Research and duplicate/overlap review come before Issue creation. If an
existing canonical Issue covers the task, use it. If no Issue exists, create
one before implementation when GitHub workflow is authorized. If Issue
creation is required but the current task does not authorize it, stop after
research and request authorization. Do not implement first and create an Issue
afterward merely to document completed work.

### Post-work Issue evidence

After authorized implementation work, post factual evidence to the canonical
Issue before handing the task back for orchestrator review. Include, as
applicable, the branch, commit SHA, PR link, files changed, validation and
concrete results, blockers or limitations, and confirmation of intentionally
unchanged scope.

Factual progress is not a semantic completion declaration. Do not close the
Issue or turn a progress comment into a project/product decision. The
owner/orchestrator retains authority over acceptance, final verdict, merge,
semantic completion, and Issue closure.

Codex may add factual engineering evidence—progress, investigation evidence,
validation results, and commit/PR links—to an existing Issue or PR during an
explicitly authorized GitHub workflow. For authorized implementation work,
the canonical Issue update described above is mandatory. Scope, decisions, and
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
