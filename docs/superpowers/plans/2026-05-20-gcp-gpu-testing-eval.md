# GCP GPU Testing Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [`docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md`](../specs/2026-05-20-gcp-gpu-testing-eval-design.md)
**Tracking issue:** [#53](https://github.com/NguyenJus/Efficient-SAM3-Finetuning/issues/53) — *Investigate Google Cloud as a GPU testing target*
**Branch:** `spec/gcp-gpu-testing-eval`

**Goal:** Execute the three-phase research pipeline defined in the spec (triage → parallel deep-dives → synthesis) and post the resulting evaluation as a single comment on issue #53. The PR landing this work contains only the spec and this plan; no `.github/workflows/`, no `docs/cloud/`, no other repo edits.

**Architecture:** Desk-research only. Phase 1 is a single serial subagent dispatch that triages all four GCP candidates against a four-disqualifier checklist and emits a smoking-gun verdict line. Phase 2 fans out one parallel subagent per Phase-1 survivor (skipped or narrowed if Phase 1 fired a smoking gun). Phase 3 synthesizes a markdown comment body (short-form, full-form, or split, per Phase-1 verdict). The orchestrator posts the comment with `gh issue comment 53 --body-file`, deletes the body file, and marks the PR ready.

**Tech Stack:** WebFetch / WebSearch (subagent tools) against GCP docs, the Cloud Billing Calculator, the Claude Code docs, and GitHub-Actions-on-GCP community write-ups. `gh issue comment` to post. No code, no YAML, no shell scripts checked in.

---

## File Map

**Committed in this PR (already present before plan execution):**

```
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md
docs/superpowers/plans/2026-05-20-gcp-gpu-testing-eval.md          # this file
```

**Created during execution but NOT committed (per spec §3):**

```
/tmp/gcp-eval-synthesis-comment.md   # the synthesis subagent's return value, materialized as a file so gh can read it
```

This file is deleted after `gh issue comment 53 --body-file <path>` succeeds (Task 6). Per-candidate Phase 1 verdicts and Phase 2 deep-dive briefs live as subagent return values inside the orchestrator session transcript — no intermediate files written to disk.

**No files modified.** No file under `.github/workflows/`, `docs/cloud/`, `notebooks/`, `src/`, `tests/`, or anywhere else.

---

## Planner-resolved decisions (locked in this plan)

The spec §10 lists seven open questions. The planner resolves them here so subagents and the orchestrator do not relitigate them:

- **OQ1 — PR-run length in the synthesis table.** **Decision:** report the **10-min worst-case** in the synthesis comparison Table A (per spec §4.3.2 item 3 axis "Cost/PR (worst — on-demand)"). Phase 2 deep-dive briefs retain **both endpoints** (5-min and 10-min) per candidate so the synthesis subagent has the underlying numbers if a candidate's recommended posture is spot-cheap-short rather than on-demand-long. Surfaces in Task 3's prompt.
- **OQ2 — Early-kill within a candidate's disqualifier check.** **Decision:** the triage subagent is **allowed to skip later disqualifier checks** once any check fires for a given candidate, **provided the verdict block names which checks were not evaluated and why**. Surfaces in Task 2's prompt and verdict-block schema.
- **OQ3 — Source of the per-PR Colab human-in-the-loop tax (§5.1 number 3).** **Decision:** the orchestrator asks the user **a single clarifying question** before dispatching the Phase 3 synthesis subagent. The question and its conditional skip-when-smoking-gun-fires-with-scope-both gate live in Task 5. The synthesis subagent receives the user's answer as part of its prompt.
- **OQ4 — Cloud Run GPU inclusion.** **Pre-resolved by the spec — no decision needed.** Cloud Run GPU stays in the candidate list. Surfaces in Task 2's input list verbatim.
- **OQ5 — PR description form.** **Decision:** the PR description is a **one-liner** that names the recommendation verdict (or both verdicts for a split case) and links to the issue comment. No comparison table in the PR description. Surfaces in Task 7. Examples: `REJECT — see #53 comment for cost/per-PR + cold-start analysis`; `REJECT for headless CI, DEFER for agentic dev — see #53 comment for analysis`.
- **OQ6 — Agentic-dev session length.** **Decision:** the brief reports a **single primary length of 2 hours** for axis 7 sub-point (d) cost figures, with a one-sentence note in the brief stating the user can scale linearly to other session lengths. Surfaces in Task 3's prompt.
- **OQ7 — Claude Code login flow.** **Decision:** the brief states the **documented browser-OAuth path** per the Claude Code docs and **flags non-trivial workarounds** (device-code flow, SSH port-forward for the OAuth redirect, copy-paste from a local login) if the documented path does not work headlessly. **Live verification is out of scope** (consistent with spec §1.3 hands-on exclusion). Surfaces in Task 3's prompt.

No plan-level open questions are left for the orchestrator to escalate.

---

## Out of scope (do not dispatch tasks for these)

Mirrors spec §9. The orchestrator must not invent tasks covering these — if a subagent return value drifts into one of these areas, the orchestrator strips it before passing forward.

- **Hands-on GCP setup** — no account conversion, no quota request, no console clicking, no live GPU smoke tests.
- **Live billing data.** All dollar figures come from published pricing pages and the Cloud Billing Calculator. No real-account spend.
- **AWS / Azure / Lambda Labs.** Tracked under #35.
- **SageMaker Studio Lab.** Tracked under #20.
- **Workflow YAML committed to the repo.** Smallest-viable-PR sketches in the synthesis comment are *illustrative fenced code blocks inside the comment body*. Nothing under `.github/workflows/` is added or edited.
- **Replacing `notebooks/esam3_train.ipynb`.**
- **Re-evaluating the GPU test policy** (assumes `2026-05-19-gpu-test-policy-design.md` is fixed).
- **Multi-region GCP cost arbitrage** — each Phase 2 brief picks one primary region per candidate (typically `us-central1`).
- **Anthropic API keys for non-Claude-Max usage.** Axis 7 assumes the owner's Claude Max OAuth login is what runs Claude Code inside the instance; a service-account API-key path is a separate question, not investigated here.

---

## Parallelization opportunities

Only **Task 4** (Phase 2 deep-dives) is parallel. Every other task is serial — Phase 1 must finish before Phase 2 (Phase 1's smoking-gun verdict gates whether Phase 2 runs at all and which axes it covers); Phase 2 must finish before Phase 3 (synthesis consumes the briefs).

```
Task 1 (pre-flight)
  → Task 2 (Phase 1, serial — 1 subagent)
    → Task 3 (Phase 2 dispatch prep)
      → Task 4 (Phase 2, parallel — 1 subagent per survivor)
        → Task 5 (pre-Phase-3 user question)
          → Task 6 (Phase 3, serial — 1 subagent)
            → Task 7 (post + close-out)
```

---

## Pre-flight check

- [ ] **Step 0a: Confirm worktree and branch**

```bash
pwd && git rev-parse --abbrev-ref HEAD
```
Expected: `/home/justin/projects/Efficient-SAM3-Finetuning/.worktrees/spec-gcp-gpu-testing-eval` and `spec/gcp-gpu-testing-eval`. If either differs, halt — the orchestrator's safety-check should have caught this in `cd`-into-worktree.

- [ ] **Step 0b: Confirm spec + plan are committed**

```bash
git log --oneline -5
git status
```
Expected: recent commits include the spec (`2026-05-20-gcp-gpu-testing-eval-design.md`) and this plan (`2026-05-20-gcp-gpu-testing-eval.md`); working tree clean.

- [ ] **Step 0c: Confirm the draft PR is open**

```bash
gh pr view --json number,state,isDraft,url --jq '{number, state, isDraft, url}'
```
Expected: `state: OPEN`, `isDraft: true`. Record the PR number for Task 7.

- [ ] **Step 0d: Confirm issue #53 is open and assignable**

```bash
gh issue view 53 --json number,state,title,url --jq '{number, state, title, url}'
```
Expected: `state: OPEN`. The synthesis comment in Task 7 lands here.

---

## Task 1: Pre-flight — confirm GCP candidate list from #53

**Goal:** Sanity-check that the four candidates named in the spec (Vertex AI Custom Jobs, Colab Enterprise, GCE + GPU spot, Cloud Run GPU) still match the canonical list in the issue body. If the issue has drifted (rare — the issue was filed recently), the spec wins per §4.1's authoritative input list; this step just surfaces the drift for the orchestrator's awareness.

**Subagent:** None — orchestrator runs this directly.

- [ ] **Step 1.1: Read the issue body**

```bash
gh issue view 53 --json body --jq .body | head -60
```

- [ ] **Step 1.2: Verify the four candidate names**

The expected names are (per spec §4.1 and §1.2): **Vertex AI Custom Jobs**, **Colab Enterprise**, **GCE + GPU spot**, **Cloud Run GPU**. If the issue lists a different fourth candidate (e.g., the issue was edited to drop Cloud Run GPU and add something else), proceed with **the spec's list** (the spec is the contract Phase 1 was designed around) and note the drift in the orchestrator's branch log as a `warn`-level entry.

- [ ] **Step 1.3: No commit — this task is read-only.**

---

## Task 2: Phase 1 — Triage pass (one subagent, serial)

**Goal:** Dispatch a single subagent that produces a per-candidate KILL/SURVIVE verdict block for all four candidates **plus** an explicit smoking-gun verdict line. This is the input to Task 4 (Phase 2 deep-dives) and gates whether Phase 2 runs at all.

**Subagent dispatch:**

- **Model/effort:** **sonnet, high.** Web-research-heavy, well-scoped, no judgment-heavy synthesis needed. The orchestrator may override to opus if Phase 1 returns ambiguous verdicts on a re-dispatch.
- **Tool budget:** WebFetch, WebSearch.
- **Output:** a single structured markdown block (returned as the subagent's reply, NOT written to a file).

- [ ] **Step 2.1: Dispatch the triage subagent**

Use the orchestrator's standard subagent-dispatch flow (Task tool or equivalent). Prompt template:

````text
You are the Phase 1 triage subagent for the GCP GPU testing evaluation
(issue #53). Spec authority:
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md §4.1.

Triage all four candidates against the four-disqualifier checklist. For
each candidate, produce a verdict block of EXACTLY this shape (no extra
keys, no extra prose outside the blocks):

```text
Candidate: <name>
Verdict: KILL | SURVIVE
Disqualifier checks performed: <list of which of the 4 disqualifiers you evaluated>
Disqualifier checks skipped: <list of disqualifiers you did NOT evaluate, with reason — e.g., "paid-account check fired hard kill; later checks moot">
Rationale: <one paragraph, ≤4 sentences>
Sources: <one or more URLs to GCP docs / community reports>
```

Early-kill is allowed: once any disqualifier fires a hard kill for a
candidate, you may skip later disqualifiers for THAT candidate provided
the "Disqualifier checks skipped" field names which were skipped and
why. (Planner decision — OQ2.)

The four candidates:
1. Vertex AI Custom Jobs
2. Colab Enterprise
3. GCE + GPU (spot)
4. Cloud Run GPU

The four disqualifiers (spec §4.1):
A. Paid-account requirement for GPU access (free-trial blocks GPU SKUs).
B. GPU quota friction (T4 / L4 / A100) — region availability + approval
   timeline for a fresh account.
C. Headless invocation from GitHub Actions — WIF (Workload Identity
   Federation) supported or practical SA-key fallback.
D. HF gated checkpoint pull — can the job pull `facebook/sam3.1` from
   huggingface.co (outbound egress, secret injection, cache surface).

Every concrete claim (quota timeline, WIF support, GPU SKU availability)
must carry a source URL.

After the four candidate verdict blocks, produce ONE additional block —
the smoking-gun verdict line per spec §4.1.1:

```text
Smoking gun: yes | no
If yes — scope: headless-CI | agentic-dev | both
If yes — rationale: <one paragraph, ≤4 sentences, with source URLs>
```

A smoking gun is any Phase 1 finding that uniformly disqualifies all
four candidates for a given use case, or makes the entire
GCP-for-this-use-case premise structurally non-viable. Indicative
examples (non-exhaustive, see spec §4.1.1): uniform paid-account-plus-
multi-week-quota across all four; structural HF egress block; uniform
absent WIF + interactive-only SA-key flow; cheapest-survivor pricing
that obviously fails (this last one is SOFT — flag as candidate, do not
declare).

Use-case-specific smoking guns are allowed (one scope value, not
"both"). If a smoking gun kills only the headless-CI use case, the
agentic-dev side may still survive — call it out.

Return ONLY the five blocks (four candidate verdicts + one smoking-gun
line). No preamble, no postscript. The orchestrator will use the blocks
verbatim as input to Phase 2 and Phase 3.

Pessimistic-prior framing (spec §2): default verdict is REJECT unless
data shows otherwise. A smoking-gun YES is a successful outcome of
triage, not a failure mode.
````

- [ ] **Step 2.2: Verify the triage subagent's return value**

Acceptance criteria (mirroring spec §4.1 pass criteria + §8 acceptance):

1. Exactly four candidate verdict blocks, one per candidate from the spec's list.
2. Each verdict is **KILL** or **SURVIVE** — no "TBD", no "needs more research".
3. Each verdict block has all five fields: `Candidate`, `Verdict`, `Disqualifier checks performed`, `Disqualifier checks skipped` (may be `none` if all four were evaluated), `Rationale`, `Sources`.
4. Each KILL block names the specific disqualifier(s) that triggered it in the `Rationale`.
5. Each SURVIVE block explicitly states that all four disqualifiers were checked and none fired (or — if early-kill was used — names which were skipped and why; for a SURVIVE, no disqualifier fired).
6. Every concrete claim (quota timeline, WIF support, SKU availability) has a source URL.
7. Exactly one smoking-gun verdict line block: `yes` or `no`, with scope and rationale if `yes`.

If any criterion fails, re-dispatch the subagent (same prompt, optionally bumping model to opus) with feedback naming the failing criterion. Up to **two re-dispatch attempts** before halting and posting the question as a draft-PR comment for the user.

- [ ] **Step 2.3: Record the verdicts in the orchestrator session**

Log the smoking-gun outcome at `info` level in `logs/spec-gcp-gpu-testing-eval.md` per CLAUDE.md log-append mechanics. One line summarizing: number of SURVIVEs, smoking-gun value (`yes` / `no` + scope if `yes`).

- [ ] **Step 2.4: No commit — verdict blocks live in session transcript only (per spec §3).**

---

## Task 3: Phase 2 dispatch prep — survivor list + axis scope

**Goal:** Compute the Phase 2 dispatch matrix from Phase 1's output. This is orchestrator bookkeeping; no subagent dispatched here.

**Subagent:** None — orchestrator runs this directly.

- [ ] **Step 3.1: Apply the skip-gate from spec §4.2**

From Phase 1's smoking-gun verdict line, determine Phase 2's shape:

| Smoking gun | Scope | Phase 2 action |
| --- | --- | --- |
| `no` | n/a | Run Phase 2 in full — axes 1–7 for each SURVIVE candidate. |
| `yes` | `both` | **Skip Phase 2 entirely.** Jump to Task 5 → Task 6 (short-form synthesis). |
| `yes` | `headless-CI` | Run Phase 2 covering **only axis 7** (agentic-dev) for each SURVIVE candidate. |
| `yes` | `agentic-dev` | Run Phase 2 covering **only axes 1–6** (headless CI) for each SURVIVE candidate. |

- [ ] **Step 3.2: Compute the survivor list**

From Phase 1's per-candidate verdict blocks, collect every candidate whose `Verdict` is `SURVIVE`. This list (possibly empty) is the parallel-dispatch matrix for Task 4.

- [ ] **Step 3.3: Handle the empty-survivor edge case**

If zero candidates survive Phase 1 but the smoking-gun verdict is `no`, this is anomalous — Phase 1 either uniformly killed all four (which should have been called as a smoking gun) or has a bug in its verdict logic. Re-dispatch Phase 1 (Task 2) with a follow-up prompt explicitly asking *"if all four candidates are killed, is there a smoking gun? If yes, classify it."* Do not proceed to Task 4 with zero survivors and no smoking gun.

- [ ] **Step 3.4: No commit — this task is bookkeeping only.**

---

## Task 4: Phase 2 — Deep-dive briefs (parallel, one subagent per survivor)

**Goal:** Dispatch N parallel subagents (N = number of Phase 1 survivors), each producing a structured deep-dive brief for one candidate. **Skipped entirely if Phase 1's smoking-gun scope is `both`** (see Task 3 skip-gate). The seven axes covered per brief depend on the smoking-gun scope per Task 3's table.

**Subagent dispatch pattern:** `superpowers:dispatching-parallel-agents`. File-disjoint by candidate (each subagent owns one candidate brief), no shared state, same branch/worktree (no `isolation: "worktree"` — would split onto temp branches).

**Per-subagent model/effort:** **sonnet, high.** Per-candidate research is bounded scope (one candidate, seven axes or a scoped subset). Orchestrator may override per task if a candidate's brief comes back thin and a re-dispatch on opus is warranted.

**Per-subagent tool budget:** WebFetch, WebSearch.

- [ ] **Step 4.1: Compose the per-candidate prompt**

The base prompt template (orchestrator substitutes `<CANDIDATE>` and the axis-scope clause per Task 3's skip-gate):

````text
You are the Phase 2 deep-dive subagent for candidate <CANDIDATE>
in the GCP GPU testing evaluation (issue #53). Spec authority:
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md §4.2.

Produce ONE structured markdown brief for <CANDIDATE> covering the
following axes:

<AXIS_SCOPE>

Where <AXIS_SCOPE> is one of:
- "all seven axes (axes 1–6 headless CI, axis 7 agentic dev with all
  five sub-points)" — if smoking-gun scope is `no`.
- "only axes 1–6 (headless CI). Skip axis 7 entirely." — if smoking-gun
  scope is `agentic-dev`.
- "only axis 7 (agentic dev with all five sub-points). Skip axes 1–6
  entirely." — if smoking-gun scope is `headless-CI`.

Brief format — one markdown section per axis, in spec-numbered order.
The seven axes (spec §4.2):

1. **Auth path from GitHub Actions** (headless CI). WIF preferred; SA
   key fallback with explicit security caveat. Cite the GCP doc page
   authorizing the preferred path.
2. **Cold-start latency for a short job** (headless CI). Target: <5 min
   from `actions/checkout` to `pytest` first line. Distinguish container-
   pull from VM-provision time when the doc allows. Sourced.
3. **Cost per PR run at T4, L4, A100** (headless CI). Spot vs.
   on-demand for each SKU. PR run = 5–10 GPU-minutes. Report rate
   ($/GPU-hour) AND per-PR cost at BOTH endpoints (5-min and 10-min) per
   SKU per spot/on-demand combination. All numbers carry "as of
   2026-05-20" + source URL. (Planner decision OQ1: both endpoints
   retained in the brief; synthesis will collapse to 10-min worst-case in
   the comment's table.)
4. **HF gated checkpoint pull viability** (headless CI). How the job
   pulls `facebook/sam3.1`: token-passing mechanism, network egress
   posture, caching options. Flag candidate-specific gotchas (e.g.,
   Cloud Run GPU's filesystem ephemerality).
5. **Dev-loop fit relative to existing Colab notebook (#48)** (headless
   CI). One of: **replaces** | **complements** | **nightly-only**. One
   short paragraph of rationale.
6. **Security posture, brief** (headless CI). Solo public project; one
   sentence summarizing axis-1 auth-path security implications.
7. **Agentic Claude Code dev fit** (agentic-dev — distinct from axes
   1–6). FIVE sub-points, one short paragraph each:

    a. **Interactive surface.** SSH, web-shell, JupyterLab terminal,
       Cloud Shell, IDE integration. Ephemeral request-response runtimes
       fail this. Cite GCP doc.

    b. **Persistent workspace.** Boot disk / persistent disk / GCS-
       mounted volume / Filestore / container image with bind-mount /
       stateless (re-clone). Cite the GCP doc for the persistence
       mechanism named.

    c. **Claude Max subscription auth path.** Can `claude` CLI
       authenticate against `api.anthropic.com` from inside the instance
       using the owner's existing Claude Max subscription? Specifically:
       (i) outbound egress to `api.anthropic.com` permitted by default;
       (ii) Claude Code login flow feasible from inside (browser-OAuth
       is the documented path per the Claude Code docs — flag non-trivial
       workarounds like device-code, SSH port-forward for OAuth redirect,
       or copy-paste of a local token; live verification is OUT OF
       SCOPE per planner decision OQ7); (iii) does session/token
       storage in `~/.claude/` survive instance restarts (depends on
       sub-point b). Cite the Claude Code docs page.

    d. **Cost dynamics for long-running agentic dev.** Cost of running
       the candidate for **2 hours/day at the cheapest GPU SKU on that
       candidate** (planner decision OQ6: single primary length of 2
       hours; include a one-sentence note that the user can scale the
       figure linearly to other session lengths). Report spot vs.
       on-demand. Compare against the Colab Free baseline ($0/session
       but T4-only and time-capped). All numbers carry "as of
       2026-05-20" + source URL.

    e. **Claude Max usage-pattern posture.** ONE short paragraph
       observing that Claude Max is a personal subscription with a
       user-bound API key / OAuth token, and that using it from a
       personal cloud VM is consistent with normal usage (the VM is the
       owner's hardware, just remote). NOT a TOS analysis; a
       usage-pattern observation.

**Sourcing requirement (spec §4.2 last paragraph + §7):** every
quantitative claim AND most qualitative claims carry a URL. GCP pricing
pages and the Cloud Billing Calculator are preferred over community
posts; community posts are acceptable for "is X feasible" but not for
dollar figures.

Pessimistic-prior framing (spec §2): the candidate is held to a high
bar against a $0/PR Colab Free baseline. Reasoned negative findings are
valuable — do not pad weak verdicts.

Return ONLY the structured markdown brief — no preamble, no
postscript. The orchestrator will pass it verbatim to the Phase 3
synthesis subagent.
````

- [ ] **Step 4.2: Dispatch all survivor subagents in parallel**

Use `superpowers:dispatching-parallel-agents` per CLAUDE.md "Parallel" guidance: same branch/worktree, no `isolation: "worktree"` (would split onto temp branches that are not needed for a research-only task). Each subagent receives its candidate-specific prompt from Step 4.1.

- [ ] **Step 4.3: Verify each brief**

Acceptance criteria per brief (mirroring spec §4.2 + §8):

1. **Axis coverage matches the smoking-gun scope** from Task 3 (full, 1–6 only, or 7 only).
2. If axis 7 is in scope, all **five sub-points** (a–e) are present and individually addressed.
3. **Every dollar figure** carries `"as of 2026-05-20"` + source URL.
4. **Every quantitative claim** (cold-start, quota timeline, $/2h-session, cost-per-PR endpoint figures) carries a source URL.
5. **Every qualitative claim** about a feature being supported (WIF, interactive surface, Claude Max auth feasibility) cites the relevant GCP or Claude Code doc page.
6. Axis 3 reports BOTH the 5-min and 10-min endpoints (per OQ1 — synthesis will collapse to 10-min in the comment table, but the brief retains both).
7. Axis 7 sub-point (d) reports a single 2-hour figure with the linear-scaling note (per OQ6).
8. Axis 7 sub-point (c) states the documented login path and flags non-trivial workarounds (per OQ7); does NOT claim to have verified the login flow live.

If a brief fails on a quantitative-claim-without-source criterion (4 or 5), re-dispatch that one candidate's subagent with feedback naming the missing source. Other criteria: judge case-by-case; minor presentation issues are orchestrator-fixable, missing axes require re-dispatch.

- [ ] **Step 4.4: Concatenate the briefs in the orchestrator session**

Hold all N briefs in the session transcript, ordered by candidate name alphabetically (deterministic, makes the synthesis prompt reproducible). This concatenation is the input to Task 6.

- [ ] **Step 4.5: No commit — briefs live in session transcript only (per spec §3).**

---

## Task 5: Pre-Phase-3 — single clarifying user question for Colab human-tax (OQ3)

**Goal:** Per OQ3, the synthesis subagent needs the owner's estimate of the per-PR human-in-the-loop tax minutes for the current Colab flow (spec §5.1 number 3). Ask the user.

**Subagent:** None — orchestrator asks the user directly.

- [ ] **Step 5.1: Determine whether the question applies**

| Smoking-gun scope | Question applicability |
| --- | --- |
| `no` | **Ask.** Full-form synthesis needs the §5.1 cost-vs-value number. |
| `agentic-dev` (headless CI survives) | **Ask.** Headless-CI side of synthesis still computes §5.1. |
| `headless-CI` (agentic dev survives) | **Skip the question.** Headless CI got short-form REJECT; §5.1 is not computed for the killed use case. The agentic-dev side does not need this number. |
| `both` | **Skip the question.** Short-form REJECT for both; no §5.1 computation. |

- [ ] **Step 5.2: Ask the user (only if Step 5.1 says "Ask")**

Post the question in-session — single line, no follow-ups:

> "For the spec's per-PR cost-vs-value framing (§5.1 number 3): roughly how many minutes per PR do you currently spend on the human-re-runs-Colab loop when the GPU smoke tests flake? A rough anecdotal estimate is fine — the synthesis comment will footnote it as your anecdotal number."

Record the user's answer verbatim. If the user does not have an estimate, accept `unknown — synthesis subagent footnotes this number as "TBD per owner"` and pass that to the synthesis subagent.

- [ ] **Step 5.3: No commit — user answer lives in session transcript only.**

---

## Task 6: Phase 3 — Synthesis comment (one subagent, serial)

**Goal:** Dispatch a single subagent that merges Phase 1 verdicts + Phase 2 briefs into the final issue-comment body. The subagent returns one markdown file path (the orchestrator materializes the file at `/tmp/gcp-eval-synthesis-comment.md`) or returns the comment body inline and the orchestrator writes it to that path. Either is fine; the file is deleted in Task 7 after `gh issue comment` succeeds.

**Subagent dispatch:**

- **Model/effort:** **opus, high.** Judgment-heavy synthesis: comparison, recommendation framing, split-verdict handling, cost-vs-value paragraph framing. Spec §4.3 is the most prose-dense part of the deliverable.
- **Tool budget:** Read (to consume the spec for §4.3 / §5 / §6 / §7 reference), Write (to materialize the comment body file).

- [ ] **Step 6.1: Choose the synthesis path from Phase 1's smoking-gun verdict**

| Smoking-gun scope | Synthesis path | Sub-step |
| --- | --- | --- |
| `no` | Full comment (spec §4.3.2) | 6.2a |
| `yes` + `both` | Short-form comment (spec §4.3.1) | 6.2b |
| `yes` + `headless-CI` | **Split comment:** short-form REJECT for headless CI + full-form content for agentic dev | 6.2c |
| `yes` + `agentic-dev` | **Split comment:** short-form REJECT for agentic dev + full-form content for headless CI | 6.2c |

- [ ] **Step 6.2a: Full comment dispatch (smoking-gun `no`)**

Prompt template:

````text
You are the Phase 3 synthesis subagent for the GCP GPU testing
evaluation (issue #53). Spec authority:
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md §4.3.2.

Phase 1 returned smoking gun: NO. Phase 2 ran in full.

INPUTS (concatenated below):
- Phase 1 verdicts (4 per-candidate blocks + smoking-gun line):
  <PASTE PHASE 1 OUTPUT>
- Phase 2 briefs (one per survivor, alphabetical):
  <PASTE PHASE 2 BRIEFS>
- Owner's per-PR Colab human-in-the-loop tax estimate (OQ3):
  <PASTE USER ANSWER FROM TASK 5, OR "unknown" IF SKIPPED>

Produce ONE markdown file body (no front-matter, no surrounding ```
fence) containing the full comment per spec §4.3.2:

1. **Preamble** — short paragraph restating the pessimistic-prior
   framing from spec §2. End with a one-line recommendation tag:
   `**Recommendation: ADOPT | DEFER | REJECT**` OR (if the two use cases
   warrant different verdicts) a split recommendation of the form
   `**Recommendation: <verdict-CI> for headless CI; <verdict-dev> for
   agentic dev**`.

2. **Candidates considered.** Bulleted list of all four candidates with
   a one-line note for any candidate killed at triage on a single
   disqualifier.

3. **Comparison tables (TWO tables, both required).**
   - **Table A — Headless CI fit.** Rows for every candidate (including
     killed). Columns per spec §4.3.2: Candidate | Cost/PR (best — spot)
     | Cost/PR (worst — on-demand) | Cold-start | GPU pinnability |
     Headless-ness | HF gated checkpoint friction | Kill reason. Use the
     10-min worst-case figure in the on-demand column (planner OQ1).
   - **Table B — Agentic dev fit.** Rows for every candidate (including
     killed). Columns per spec §4.3.2: Candidate | Interactive surface |
     Persistent workspace | Claude Max auth | $/2h-session (best — spot)
     | $/2h-session (worst — on-demand) | Kill reason.

4. **Cost-vs-value lens (spec §5).** TWO short paragraphs, one per use
   case.
   - **Headless CI (§5.1):** explicitly compute three numbers — best/
     worst $/PR for the recommended candidate (or cheapest survivor),
     setup tax in hours, per-PR human-in-the-loop tax in minutes (from
     OQ3 input above). Frame as: "GCP costs $X-Y/PR but eliminates Z
     minutes/PR of human-re-run-Colab toil, against a one-time setup tax
     of T hours. At the current PR cadence of ~N PRs/month, breakeven on
     the setup tax is M months." If the owner's per-PR tax estimate is
     "unknown", footnote it as TBD-per-owner and frame the trade-off
     conditionally.
   - **Agentic dev (§5.2):** explicitly compute three numbers — best/
     worst $/2h session for the recommended candidate on its cheapest
     GPU SKU, setup tax in hours, and the per-session value (capability
     gap — L4/A100 codepaths the Colab T4 cannot exercise per #9). Frame
     as: "GCP costs $X-Y per 2-hour session and unlocks L4/A100-class
     codepaths the Colab T4 cannot exercise (#9), against a setup tax of
     T hours (including Claude Code install + Claude Max login). The
     Colab Free baseline remains $0/session but T4-only and time-capped."

5. **Recommendation block.** Addresses BOTH use cases. May be single or
   split verdict.
   - If ADOPT (either or both): smallest-viable-PR sketch in fenced code
     blocks (Vertex Custom Job YAML / WIF steps / draft workflow YAML
     skeleton for headless CI; instance-creation sketch / Claude Code
     install+login sequence / workspace-persistence sketch for agentic
     dev). NOTHING IS COMMITTED — these are illustrative blocks inside
     the comment body.
   - If DEFER (either or both): "what would change our mind" clause with
     at least ONE quantitative/observable trigger per deferred use case.
   - If REJECT (either or both): same as DEFER but stronger framing —
     re-evaluation expected only on a structural GCP change, not a price
     drift.

6. **Price-drift caveat.** One sentence: all dollar figures are as of
   2026-05-20 and require re-validation if the issue is revisited.

7. **Cross-references footer.** "Related issues" block with the one-line
   notes from spec §6, verbatim:
   - #48 — current Colab GPU test loop; baseline being undercut.
   - #44 — manual GPU pass cadence; per-PR human-in-the-loop tax source.
   - #9 — sm_75+ codepaths needing real-hardware coverage; motivates
     "complements" verdict (headless CI) and per-session value (agentic
     dev).
   - #35 — AWS / Lambda Labs investigation; explicitly out of scope.
   - #20 — SageMaker Studio Lab; explicitly out of scope.

Target comment length: roughly 400–800 lines of markdown. Long enough
to carry both tables + per-candidate rationales; short enough that the
recommendation is visible without scrolling past three screens.

Tone: matter-of-fact, sourced, no marketing language. The pessimistic-
prior framing means a REJECT or split DEFER/REJECT is a valid outcome,
not a failure mode.

Return the comment body as either (a) the full markdown text in your
reply, which the orchestrator writes to /tmp/gcp-eval-synthesis-comment.md;
or (b) a tool call that writes the file directly to that path. Either
works.
````

- [ ] **Step 6.2b: Short-form comment dispatch (smoking-gun `yes` + scope `both`)**

Prompt template:

````text
You are the Phase 3 synthesis subagent for the GCP GPU testing
evaluation (issue #53), short-form path. Spec authority:
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md §4.3.1.

Phase 1 fired a smoking gun with scope `both`. Phase 2 was skipped.

INPUTS:
- Phase 1 verdicts (4 per-candidate blocks + smoking-gun line):
  <PASTE PHASE 1 OUTPUT>

Produce ONE markdown file body (no front-matter, no fence) containing
the short-form comment per spec §4.3.1:

1. **Preamble** — short paragraph restating the pessimistic-prior
   framing from §2 AND naming the use-case scope (`both`). End with a
   one-line tag: `**Recommendation: REJECT (use case: both)**`.

2. **Smoking-gun finding** — one short paragraph, ≤4 sentences, naming
   the disqualifier that uniformly killed all four candidates (or the
   structural GCP-for-this-use-case blocker), with source URLs from
   Phase 1.

3. **Degenerate comparison table (optional).** A single-column-of-
   content table with rows for all four candidates and a shared
   "killed by smoking gun: <reason>" cell, OR omit entirely if it adds
   no information. Pick whichever is clearer.

4. **One-line REJECT recommendation** for both use cases with a "what
   would change our mind" trigger — at least one observable /
   quantitative trigger.

5. **Price-drift caveat** — same one-sentence reminder.

6. **Cross-references footer** — same #48, #44, #9, #35, #20 block as
   the full comment, with the spec §6 one-line notes verbatim.

Target length: roughly 80–200 lines of markdown.

Return the comment body in your reply or as a direct file write to
/tmp/gcp-eval-synthesis-comment.md.
````

- [ ] **Step 6.2c: Split comment dispatch (smoking-gun `yes` + scope `headless-CI` or `agentic-dev`)**

Prompt template:

````text
You are the Phase 3 synthesis subagent for the GCP GPU testing
evaluation (issue #53), SPLIT path. Spec authority:
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md §4.3
(combination of §4.3.1 for the killed use case and §4.3.2 for the
surviving use case).

Phase 1 fired a smoking gun with scope `<KILLED_SCOPE>` (one of
`headless-CI` or `agentic-dev`). Phase 2 ran in narrowed form, covering
only the surviving use case's axes for each Phase-1 survivor.

INPUTS:
- Phase 1 verdicts (4 per-candidate blocks + smoking-gun line):
  <PASTE PHASE 1 OUTPUT>
- Phase 2 narrowed briefs (one per survivor):
  <PASTE PHASE 2 BRIEFS>
- Owner's per-PR Colab human-in-the-loop tax estimate (if surviving
  use case is headless-CI; else "n/a — agentic dev survives"):
  <PASTE USER ANSWER OR "n/a">

Produce ONE markdown file body containing a SPLIT comment:

**Part A — short-form REJECT for the killed use case** (per §4.3.1):
   1. Preamble naming the killed scope. One-line tag:
      `**Recommendation: REJECT (use case: <KILLED_SCOPE>)**`.
   2. Smoking-gun finding — one short paragraph with source URLs.
   3. Optional degenerate table.
   4. One-line REJECT with "what would change our mind" trigger.

**Part B — full-flow content for the surviving use case** (per §4.3.2,
   adapted — only the surviving use case's content):
   - Surviving-use-case preamble. One-line tag:
     `**Recommendation: <ADOPT | DEFER | REJECT> (use case:
     <SURVIVING_SCOPE>)**`.
   - Candidates considered (same four).
   - Only the relevant comparison table (Table A if headless CI
     survives, Table B if agentic dev survives), with rows for every
     candidate.
   - Cost-vs-value lens for the surviving use case only (one paragraph
     from §5.1 or §5.2 — not both).
   - Recommendation block for the surviving use case only.

**Shared footers (one of each, not duplicated):**
   - Price-drift caveat.
   - Cross-references footer with spec §6 one-line notes.

**Combined preamble at the top** (above Parts A and B) names BOTH
verdicts in one tag, e.g.:
   `**Recommendation: REJECT for headless CI, DEFER for agentic dev**`

Target length: roughly 200–500 lines (between short-form and full-flow
targets, since one use case is short-form and one is full).

Return the comment body in your reply or via direct file write to
/tmp/gcp-eval-synthesis-comment.md.
````

- [ ] **Step 6.3: Materialize the comment body file**

If the subagent returned the body inline, the orchestrator writes it:

```bash
cat > /tmp/gcp-eval-synthesis-comment.md <<'EOF'
<paste synthesis subagent's return value here>
EOF
```

If the subagent wrote the file directly, verify it exists:

```bash
test -s /tmp/gcp-eval-synthesis-comment.md && echo "OK" || echo "MISSING"
```

Expected: `OK`.

- [ ] **Step 6.4: Verify the comment body against spec §8 acceptance criteria**

For the **full-form path** (Step 6.2a) — check every criterion:

1. Preamble restates pessimistic-prior framing + one-line recommendation tag.
2. Bulleted list of all four candidates with one-line kill note for any killed-at-triage candidate.
3. **Table A — Headless CI fit** is present with rows for ALL four candidates and the exact column set from spec §4.3.2 (Candidate, Cost/PR best-spot, Cost/PR worst-on-demand, Cold-start, GPU pinnability, Headless-ness, HF friction, Kill reason). Killed candidates show `n/a` in cost columns.
4. **Table B — Agentic dev fit** is present with rows for ALL four candidates and the exact column set (Candidate, Interactive surface, Persistent workspace, Claude Max auth, $/2h spot, $/2h on-demand, Kill reason). Killed candidates show `n/a` in interior cells.
5. Two cost-vs-value paragraphs (one per use case) with the three numbers per use case computed in plain prose.
6. Recommendation block addresses BOTH use cases (single or split verdict). Verdict is one of ADOPT / DEFER / REJECT for each use case.
7. If any verdict is ADOPT: smallest-viable-PR sketch present as fenced code blocks. **Verify NOTHING under `.github/workflows/` is being added — the sketches are inside the comment body only.**
8. If any verdict is DEFER or REJECT: at least one quantitative/observable trigger per deferred or rejected use case.
9. Every dollar figure has `"as of 2026-05-20"` + source URL.
10. Every quantitative claim has a source URL.
11. Cross-references footer lists #48, #44, #9, #35, #20 with spec §6 one-liners.
12. Length is roughly 400–800 lines.

For the **short-form path** (Step 6.2b) — check:

1. Preamble restates pessimistic-prior framing + names use-case scope + one-line REJECT tag.
2. Smoking-gun finding paragraph with source URLs.
3. One-line REJECT + observable trigger.
4. Price-drift caveat + cross-references footer.
5. Length is roughly 80–200 lines.

For the **split path** (Step 6.2c) — check both subsets above, restricted to the relevant use cases.

If any criterion fails, re-dispatch the subagent (same model/effort) with feedback naming the failing criterion. Up to **two re-dispatch attempts** before halting and escalating to the user.

- [ ] **Step 6.5: No commit — the comment body file lives at `/tmp/gcp-eval-synthesis-comment.md`, untracked.**

---

## Task 7: Post the comment + close-out

**Goal:** Post the synthesis comment to issue #53, delete the temporary file, update the PR description with the one-line recommendation, and prepare for orchestrator-standard CI-watch + PR-ready flow.

**Subagent:** None — orchestrator runs this directly.

- [ ] **Step 7.1: Post the comment**

```bash
gh issue comment 53 --body-file /tmp/gcp-eval-synthesis-comment.md
```

Expected: a `https://github.com/.../issues/53#issuecomment-<id>` URL printed to stdout. Record the URL for the PR description.

- [ ] **Step 7.2: Verify the comment posted**

```bash
gh issue view 53 --comments --json comments --jq '.comments[-1] | {url: .url, createdAt: .createdAt, bodyLength: (.body | length)}'
```

Expected: the most recent comment matches what was just posted (length within ~5% of `/tmp/gcp-eval-synthesis-comment.md`'s line count, timestamp within the last minute).

- [ ] **Step 7.3: Delete the temporary comment-body file (spec §3 requirement)**

```bash
rm /tmp/gcp-eval-synthesis-comment.md
test ! -e /tmp/gcp-eval-synthesis-comment.md && echo "deleted" || echo "STILL PRESENT"
```

Expected: `deleted`. The comment body remains in the orchestrator session transcript (synthesis subagent's return value) and on the issue itself; the local file is not needed and is not committed.

- [ ] **Step 7.4: Update the PR description with the one-line recommendation (OQ5)**

Compose the one-liner per planner decision OQ5. Examples:

- Single verdict (smoking-gun no, both use cases same verdict): `REJECT — see #53 comment for cost/per-PR + cold-start analysis`.
- Smoking-gun both: `REJECT (both use cases) — see #53 comment for smoking-gun rationale`.
- Split: `REJECT for headless CI, DEFER for agentic dev — see #53 comment for analysis`.

Update the PR description, preserving any "Closes #53" line the draft PR already has (or add one if missing — the PR must close #53 on merge):

```bash
gh pr edit --body "$(cat <<'EOF'
<one-line recommendation per OQ5>

See the [issue #53 evaluation comment](<URL from Step 7.1>) for the full analysis (sourced cost figures, comparison tables, cost-vs-value framing, and recommendation block).

Closes #53
EOF
)"
```

- [ ] **Step 7.5: Confirm spec + plan are the only files in the diff**

```bash
gh pr diff --name-only
```

Expected output: exactly two files:

```
docs/superpowers/plans/2026-05-20-gcp-gpu-testing-eval.md
docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md
```

If any other file appears in the diff, halt — the spec §8 acceptance criterion "the PR landing this work contains only the spec and the implementation plan under `docs/superpowers/{specs,plans}/`" has been violated. Investigate which task introduced the stray file and undo.

- [ ] **Step 7.6: Mark PR ready and watch CI (standard orchestrator close-out)**

Per CLAUDE.md "Implementation-Orchestrator Pipeline step 4" — no 4-option menu. Mark the draft PR ready and watch CI. Note from project memory: `gh pr ready` doesn't re-fire skipped draft CI runs; force a push (or merge from main) before watching CI:

```bash
git commit --allow-empty -m "ci: kick"
git push
gh pr ready
```

Then watch CI without polling sleeps (use `run_in_background` / Monitor per CLAUDE.md). Notify the user only when CI is green. On failure, fix and re-loop before notifying.

- [ ] **Step 7.7: Branch close-out on merge** (handled by orchestrator's standard close-out, NOT this plan)

Per CLAUDE.md "Implementation-Orchestrator Pipeline step 5" — on user-merge, the orchestrator kills background processes, folds the branch log into `logs/logs.md`, removes the worktree, and signs off. This plan does not enumerate those steps because they are universal across plans.

---

## Acceptance Criteria (mirrors spec §8; orchestrator verifies before declaring complete)

1. Phase 1 produced a KILL-or-SURVIVE verdict for all four candidates with sourced rationale; no "TBD" entries. *(Task 2.2.)*
2. Phase 1 produced an explicit smoking-gun verdict line per spec §4.1.1. *(Task 2.2.)*
3. **Branching:**
   - If smoking-gun scope `both`: short-form comment posted per spec §4.3.1. Criteria 4, 5, 6, 7 below do NOT apply.
   - If smoking-gun scope is one use case only: split comment posted (short-form REJECT for killed + full-form for surviving). Criteria 4–7 below apply only to the surviving use case's content.
   - If smoking-gun `no`: full comment posted per spec §4.3.2. All criteria below apply. *(Task 6.1, 6.4, 7.1.)*
4. **Normal flow:** Phase 2 produced one deep-dive brief per Phase 1 survivor covering all seven axes (or narrowed scope per the smoking-gun gate), parallel-dispatchable, no cross-candidate dependencies. *(Task 4.)*
5. **Axis 7 coverage:** every surviving candidate's brief covers all five sub-points (a–e) — UNLESS the agentic-dev use case was killed by a use-case-specific smoking gun. *(Task 4.3.)*
6. **Normal flow:** the comment contains BOTH comparison tables (Table A and Table B) with rows for every candidate, the cost-vs-value paragraphs (one per use case with the three §5 numbers), and a recommendation block addressing both use cases. *(Task 6.4.)*
7. The recommendation is one of ADOPT / DEFER / REJECT (or a split of two such verdicts). *(Task 6.4.)*
8. If ADOPT: smallest-viable-PR sketches present as fenced code blocks in the comment body. No file under `.github/workflows/` is added. *(Task 6.4 item 7; Task 7.5.)*
9. If DEFER or REJECT: at least one observable/quantitative trigger in the "what would change our mind" clause per deferred/rejected use case. *(Task 6.4.)*
10. Every dollar figure carries `"as of 2026-05-20"` + source URL. *(Task 4.3, Task 6.4.)*
11. Every candidate-specific quantitative claim carries a source URL. *(Task 4.3.)*
12. Cross-references #48, #44, #9, #35, #20 appear in the comment with spec §6 one-liners. *(Task 6.4.)*
13. The PR diff contains only the spec and the plan under `docs/superpowers/{specs,plans}/`. *(Task 7.5.)*
14. Issue #53 closes when the PR merges (via "Closes #53" in the PR description + the synthesis comment). *(Task 7.4.)*
15. The PR description's one-line recommendation reflects the short-form-vs-full-vs-split path Phase 3 took (planner OQ5). *(Task 7.4.)*
