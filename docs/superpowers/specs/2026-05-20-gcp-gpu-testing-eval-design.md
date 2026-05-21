# GCP GPU Testing Evaluation Design (issue #53)

**Status:** Draft (2026-05-20)
**Tracking issue:** #53
**Scope:** Produce a structured, sourced evaluation of Google Cloud as a headless GPU CI / dev-loop target for this repo's `pytest -m gpu` suite. The final deliverable is **a single comment on issue #53** — not a checked-in `docs/` page. The PR lands only this spec and its implementation plan; the comment is posted by the orchestrator at merge time and closes #53.

**Frame:** The owner holds a **pessimistic prior on GCP** going in (see §2). The evaluation must disconfirm that prior with sourced data, not start neutral. A reasoned "defer" or "reject" outcome is fully acceptable — possibly the expected one.

---

## 1. Goals & Scope

Resolve issue #53 by producing the five-section evaluation the issue asks for, structured so the recommendation is grounded in concrete cost-per-PR dollar figures and per-candidate kill criteria rather than vibes. The execution is a three-phase research pipeline (triage → parallel deep-dives → synthesis) culminating in one issue comment.

The evaluation covers **two distinct use cases** that overlap on candidates but differ on requirements: **(a) headless `pytest -m gpu` CI driven by GitHub Actions** (ephemeral jobs, cost-per-PR, cold-start matters, no interactive surface needed), and **(b) agentic Claude Code dev with the owner's existing Claude Max subscription** (long-running or fast-warm instance, persistent workspace, interactive shell, outbound egress to `api.anthropic.com`, Claude Code OAuth login feasible from inside the instance). Each candidate is scored against both use cases; the recommendation block may issue a split verdict (e.g., REJECT for headless CI, DEFER for agentic dev) if the data warrants it.

### 1.1 Issue absorbed and closed by this PR

| Issue | Title (short) | Disposition |
| --- | --- | --- |
| #53 | Investigate Google Cloud as a GPU testing target | Closed by the synthesis comment this PR's plan produces |

### 1.2 In scope

- A **Phase 1 triage pass** screening all four candidates (Vertex AI Custom Jobs, Colab Enterprise, GCE + GPU spot, Cloud Run GPU) against a cheap disqualifier checklist; output is a per-candidate kill / survive verdict with sourced rationale, plus an explicit smoking-gun verdict line (see §4.1) that can short-circuit the pipeline.
- **Phase 2 deep-dive briefs**, one per survivor, in parallel; each brief covers the seven axes named in §4.2. Axes 1–6 cover the **headless CI** use case; axis 7 covers the **agentic Claude Code dev** use case.
- An assessment of the candidate's fit for **running Claude Code agentically** inside it using the owner's Claude Max subscription — interactive surface, persistent workspace, Claude Max auth path, agentic-dev cost dynamics, and Claude Max usage-pattern posture (Phase 2 axis 7).
- A **Phase 3 synthesis comment** posted to issue #53, containing: two comparison tables (one for headless CI fit, one for agentic dev fit) covering every candidate (including killed ones), an explicit cost-vs-value lens against the $0/PR Colab Free baseline (for both use cases), an adopt/defer/reject recommendation that may be split per use case, and either smallest-viable-PR sketches (if adopt) or a "what would change our mind" clause (if defer/reject).
- A **short-form synthesis path** for the fail-fast case where Phase 1 produces a smoking-gun verdict (see §4.1, §4.3); this is a successful outcome of the eval, not a failure mode.
- Concrete dollar-cost numbers — best-case spot and worst-case on-demand for headless CI, plus a cost-per-2-hour-agentic-session figure for agentic dev — annotated **"as of 2026-05-20"** with a source URL.
- Cross-references to related issues (#48, #44, #9, #35, #20) where they bear on a phase's reasoning.

### 1.3 Out of scope (explicitly deferred)

- **Hands-on GCP setup** — no account conversion, no quota request, no live console clicking, no smoke tests against real GCP hardware. Investigation only. (The issue says so explicitly.)
- **Live billing data.** Numbers come from published pricing pages and Cloud Billing Calculator estimates, not from a real GCP account. The synthesis comment must say this in plain language and flag that prices drift.
- **AWS / Azure / Lambda Labs.** Tracked separately under #35.
- **SageMaker Studio Lab.** Tracked separately under #20.
- **Workflow YAML files actually committed to the repo.** Any `.github/workflows/gpu-tests.yml`, Vertex YAML, or WIF configuration that appears in the synthesis comment is a *sketch* — written in fenced code blocks inside the comment body. Nothing under `.github/workflows/` is added or edited by this PR.
- **Replacing the user-facing Colab beginner notebook** `notebooks/esam3_train.ipynb` — out of scope per the issue.
- **Re-evaluating the GPU test policy itself.** The evaluation assumes the tier policy from `2026-05-19-gpu-test-policy-design.md` (specifically the `pytest -m gpu` invocation against the `gpu_smoke_lora.yaml` / `gpu_smoke_qlora.yaml` configs); it does not propose changes to that policy.

---

## 2. Pessimistic-Prior Framing (load-bearing)

The owner's prior on GCP is **pessimistic**, for concrete reasons:

- **Free-trial credit blocks GPU SKUs.** GCP advertises $300 in free-trial credit, but Google explicitly prohibits GPU usage on free-trial accounts. To run a single GPU job, the account must convert to a paid (billing-enabled) account, at which point the $300 credit is supplemental rather than free.
- **Console-exploration friction.** The owner's initial GCP console exploration — quota pages, project setup, region picker, IAM — proved to be a nuisance relative to the comparable RunPod / Colab flow.
- **Cost-zero baseline.** The current Colab Free loop costs **$0/PR**. Any GCP solution that is not free starts the comparison at a strict cost disadvantage; the dev-loop value has to clear that cost bar.

This frames the evaluation asymmetrically:

- The default verdict is **reject** unless the data shows otherwise.
- Each phase is asking *"does this candidate / GCP-in-general clear a high bar?"* — not *"is this candidate good?"*
- The synthesis recommendation is allowed to be a one-line *"reject for now, here is the trigger that would flip this"*. That is a successful outcome of the evaluation, not a failure.
- The pessimistic prior applies to **both** use cases (headless CI **and** agentic Claude Code dev) — the owner has held both questions in mind from the start, and each must independently clear the bar.
- Fast-rejection on a **smoking gun** surfaced during Phase 1 (see §4.1) is also an *explicit, valued* outcome — it saves orchestrator effort and the owner's read-time. A smoking gun may disqualify both use cases or only one; the synthesis adjusts accordingly (see §4.3).

The synthesis comment must restate this framing in its preamble so a future reader (the owner re-reading the comment six months later, or anyone discovering the issue) understands the bar the evaluation was held to.

---

## 3. Deliverable Shape

The final deliverable is **one comment on issue #53**, posted by the implementation orchestrator using `gh issue comment 53 --body-file <path>`. The synthesis subagent's job (§4.3) is to produce the comment body as a markdown file the orchestrator then posts.

The PR (this spec + the plan) lands only:

- `docs/superpowers/specs/2026-05-20-gcp-gpu-testing-eval-design.md` (this file).
- `docs/superpowers/plans/2026-05-20-gcp-gpu-testing-eval.md` (the planner subagent will write this).

The PR description summarizes the recommendation in one line (adopt / defer / reject, or a split verdict per use case — e.g., "REJECT for headless CI, DEFER for agentic dev") and links to the issue comment. On merge, the synthesis comment is posted and issue #53 closes.

**Nothing else is checked in.** Phase 1's per-candidate triage notes, the Phase 2 deep-dive briefs, and any intermediate research scratch live inside the orchestrator's session (as subagent return values that the orchestrator stitches together) — not as files in the repo. Only the final comment body file is created, and that file is deleted after `gh issue comment` succeeds. The synthesis subagent's return value contains the full comment text so it remains in the session transcript regardless.

---

## 4. Execution Pipeline — Triage, Deep-Dive, Synthesize

Three phases. Each phase is one or more discrete subagent dispatches. The implementation plan (separate doc) specifies the exact `Task` invocations; this spec defines the *contract* of each phase.

### 4.1 Phase 1 — Triage pass (one subagent, serial)

**Goal:** Eliminate candidates that fail a cheap disqualifier without spending deep-dive effort on them. Surface concrete reasons for the kill so the killed candidate still appears in the §4.3 comparison table.

**Input:** The four candidates named in #53 — Vertex AI Custom Jobs, Colab Enterprise, GCE + GPU (spot), Cloud Run GPU.

**Disqualifier checklist (each candidate scored against all four):**

| Disqualifier | What "fail" looks like | Why it matters |
| --- | --- | --- |
| **Paid-account requirement for GPU access** | Candidate requires a billing-enabled (non-free-trial) account *and* an additional quota-increase request for any GPU SKU we'd use (T4, L4, A100). | Confirms or refutes the owner's free-trial-block experience for this candidate. A "yes, paid account required AND multi-day quota approval" is a soft kill on its own; a hard kill if combined with another disqualifier. |
| **GPU quota friction (T4 / L4 / A100)** | Region availability and typical approval timeline for a fresh account; any prerequisites (e.g., spend history, billing verification). | Even if the SKU exists, a multi-week quota approval makes the candidate unusable for a "land this week" PR. |
| **Headless invocation from GitHub Actions** | WIF (Workload Identity Federation) is not supported or not practical for this candidate; SA-key fallback is not viable (e.g., service requires interactive console steps). | A GitHub-Actions-driven CI path requires headless, keyless-or-rotatable auth. WIF is the gold standard; an SA key is acceptable but flagged. No path here = hard kill. |
| **HF gated checkpoint pull** | The candidate cannot pull `facebook/sam3.1` from huggingface.co inside the job — e.g., no outbound internet, no way to inject `HF_TOKEN` as a secret, no cache surface. | The test suite is meaningless without the real checkpoint. No path here = hard kill. |

**Output:** A per-candidate verdict block of the form:

```text
Candidate: <name>
Verdict: KILL | SURVIVE
Rationale: <one paragraph, ≤4 sentences>
Sources: <one or more URLs to GCP docs / community reports>
```

The triage subagent runs `WebFetch` / `WebSearch` against GCP documentation pages, the Cloud Billing Calculator, GitHub-Actions-on-GCP community write-ups, and similar sources. Every concrete claim (quota timeline, WIF support, GPU SKU availability in a region) carries a source URL in the rationale.

**4.1.1 Smoking-gun detection (fail-fast clause).**

In addition to per-candidate verdicts, the triage subagent must produce an explicit **smoking-gun verdict line** in its return value. A *smoking gun* is any Phase 1 finding that *uniformly disqualifies all four candidates* for a given use case, OR makes the entire GCP-for-this-use-case premise structurally non-viable — i.e., a finding that lets a reasonable reader, given only the Phase 1 verdict block, conclude "no point doing the deep-dive — the recommendation is already known." The orchestrator is the final judge of whether the test is met; the triage subagent's role is to surface the candidate finding and label it.

The test is a *test*, not a closed list. Indicative examples (non-exhaustive):

- All four candidates require paid-account conversion + multi-week quota approval for any usable GPU SKU.
- GCP egress policy makes pulling `facebook/sam3.1` from `huggingface.co` structurally impossible inside any of the four candidates.
- WIF support is uniformly absent across all four candidates *and* the SA-key fallback would require interactive console steps for every candidate.
- Cheapest-survivor pricing on the headless-CI axis blows past a "would obviously not adopt at any cost" line (e.g., >$5/PR best-case spot). This one is **soft** — the triage subagent flags it as a *candidate* smoking gun; the synthesis subagent or orchestrator decides whether it rises to a smoking gun or is just a normal "reject" recommendation.
- For the agentic-dev use case specifically: no candidate provides any interactive surface on a GPU-equipped instance; or the Claude Code OAuth login flow is structurally infeasible from inside every candidate.

**Use-case-specific smoking guns are allowed.** A smoking gun may kill the headless-CI use case while leaving the agentic-dev use case alive (or vice versa). In that case Phase 2 still runs for the surviving use case (the deep-dive briefs cover only the relevant axes — axes 1–6 if headless CI survives, axis 7 if agentic dev survives), and Phase 3 produces a *split* comment: short-form REJECT for the killed use case, normal-flow recommendation for the surviving one. The triage subagent's verdict line names *which* use case is killed.

**Verdict line format** (returned alongside the four per-candidate verdicts):

```text
Smoking gun: yes | no
If yes — scope: headless-CI | agentic-dev | both
If yes — rationale: <one paragraph, ≤4 sentences, with source URLs>
```

**Pass criteria for the triage phase:**

- All four candidates have an explicit verdict (no "TBD", no "needs more research" — escalate as an open question if genuinely undecidable).
- Each KILL verdict names the specific disqualifier(s) that triggered it.
- Each SURVIVE verdict explicitly states that none of the four disqualifiers fired (with source URLs proving the negatives).
- The return value includes the explicit **smoking-gun verdict line** described in §4.1.1 — `yes` or `no`, with scope and rationale if `yes`.
- The verdict list is the input to both Phase 2 (which only deep-dives survivors, and only for the surviving use case if a use-case-specific smoking gun fired) and Phase 3 (which surfaces killed candidates in the comparison table with their kill reason, or takes the short-form comment path if a smoking gun fired).

### 4.2 Phase 2 — Deep-dive pass (one subagent per survivor, parallel)

**Skip gate.** Phase 2 is **skipped entirely** if Phase 1's smoking-gun verdict is `yes` with scope `both`. If the smoking-gun scope is `headless-CI` only, Phase 2 runs but covers only axis 7 (agentic dev) for each survivor. If the smoking-gun scope is `agentic-dev` only, Phase 2 runs but covers only axes 1–6 (headless CI) for each survivor. If the smoking-gun verdict is `no`, Phase 2 runs in full as described below.

**Goal:** For each candidate that survived triage, produce a brief covering the seven deliverable axes — six framed around the **headless CI** use case (axes 1–6, anchored on the issue's original asks), one framed around the **agentic Claude Code dev** use case (axis 7). Dispatched in parallel via the `superpowers:dispatching-parallel-agents` pattern (file-disjoint by candidate; no shared state).

**Per-candidate brief axes:**

1. **Auth path from GitHub Actions.** *(Headless CI.)* WIF preferred (describe the trust-relationship setup, the OIDC provider config, the IAM-binding step at a sketch level). SA key only as a fallback, with an explicit security caveat ("key material is a long-lived credential and must be rotated; less defensible than WIF for a public repo"). Cite the GCP doc page that authorizes the preferred path.
2. **Cold-start latency for a short job.** *(Headless CI.)* Target: <5 min from `actions/checkout` to `pytest` first line. Sourced from GCP docs and credible community reports (blog posts, talks, GitHub repos demonstrating GitHub Actions → GCP GPU jobs). Distinguish container-pull time from VM-provision time when the doc allows.
3. **Cost per PR run at T4, L4, A100.** *(Headless CI.)* **Spot vs. on-demand** for each SKU. Assume a "PR run" is **5–10 GPU-minutes** (the rough length of `pytest -m gpu` against `gpu_smoke_lora.yaml` and `gpu_smoke_qlora.yaml` per `2026-05-19-gpu-test-policy-design.md`). The brief reports the rate ($/GPU-hour) and the resulting per-PR cost at both endpoints of the 5–10 minute window. All numbers carry **"as of 2026-05-20"** annotation and a source URL.
4. **HF gated checkpoint pull viability.** *(Headless CI.)* How the job pulls `facebook/sam3.1`: token-passing mechanism (GCP Secret Manager vs. plain GitHub Actions secret env), network egress posture (any restrictions on outbound to huggingface.co), and caching options (e.g., reusable disk image, Cloud Storage staging). Flag any candidate-specific gotcha (e.g., Cloud Run GPU's filesystem ephemerality).
5. **Dev-loop fit relative to the existing Colab notebook (#48).** *(Headless CI.)* One of three verdicts: **replaces** (this candidate could obsolete the per-branch Colab notebook), **complements** (e.g., L4/A100 here for the sm_80+ codepaths the T4 can't exercise per #9, while Colab remains the default), or **nightly-only** (this candidate is too expensive or too slow per run to gate every PR, but could run as a scheduled sweep). One short paragraph of rationale.
6. **Security posture (brief).** *(Headless CI.)* Solo public project; not the main axis. Note the auth-path security implications from axis 1 in one sentence — e.g., "WIF means no long-lived key in GitHub Actions secrets, which is the right posture for a public repo's CI."
7. **Agentic Claude Code dev fit.** *(Agentic-dev use case — distinct from axes 1–6.)* This axis is framed around the owner running the `claude` CLI agentically inside the candidate, authenticated against `api.anthropic.com` via the owner's existing **Claude Max** subscription, to iterate on GPU code on the remote GPU-equipped environment. The brief covers five sub-points, one short paragraph each:

    a. **Interactive surface.** Does the candidate provide a way to run an interactive shell on a GPU-equipped instance? SSH, web-shell, JupyterLab terminal, Cloud Shell, IDE integration — any of these counts. Ephemeral request-response runtimes (e.g., Cloud Run GPU's HTTP model) likely fail this sub-point. Cite the GCP doc page that documents the interactive path.

    b. **Persistent workspace.** Can the owner's `Efficient-SAM3-Finetuning` working directory persist across sessions so Claude Code can resume work? Options: boot disk, persistent disk, GCS-mounted volume (`gcsfuse`), Filestore, container image with bind-mount. Or is the instance fundamentally stateless (re-clone on every session)? Cite the GCP doc page for the persistence mechanism named.

    c. **Claude Max subscription auth path.** Can `claude` CLI authenticate against `api.anthropic.com` from inside the instance using the owner's existing Claude Max subscription, the same way it does from a local laptop? Specifically: (i) outbound egress to `api.anthropic.com` permitted by default; (ii) the Claude Code login flow (browser-based OAuth) feasible from inside the instance, or does it require local-browser steps; (iii) does the session/token storage in `~/.claude/` survive instance restarts (depends on persistent-workspace choice from sub-point b). Cite the Claude Code docs page for the relevant login mechanics.

    d. **Cost dynamics for long-running agentic dev.** Different from per-PR cost. Calculate: cost of running the candidate for **2 hours/day at the cheapest GPU SKU on that candidate** (rough proxy for an agentic-dev session). Report spot vs. on-demand. Compare against the Colab Free baseline ($0/session, but T4-only and time-capped). All numbers carry **"as of 2026-05-20"** annotation and a source URL.

    e. **Claude Max usage-pattern posture.** One short paragraph. Note that Claude Max is a personal subscription but the auth surface is a user-bound API key / OAuth token — using it from a personal cloud VM is consistent with normal usage (the VM is the owner's hardware, just remote). Flag this in plain language so the owner has the framing in writing. This is *not* a TOS analysis ("is this allowed by Anthropic?"); it is a usage-pattern observation ("is the auth model designed for this usage pattern?").

**Format:** Each brief is returned by its subagent as a structured markdown block. The orchestrator concatenates them as input to Phase 3; they are not committed to the repo.

**Sourcing requirement:** Every quantitative claim (cost figure, cold-start number, quota timeline, 2-hour-session cost, interactive-surface availability, Claude Max auth feasibility) carries a URL. Qualitative claims (e.g., "WIF is supported on Vertex Custom Jobs", "the candidate exposes an SSH surface", "Claude Code's OAuth flow works headlessly") cite the relevant GCP or Claude Code doc page.

### 4.3 Phase 3 — Synthesis pass (one subagent, serial)

**Goal:** Merge all Phase 1 verdicts and Phase 2 briefs into the final issue comment body. Returns one markdown file ready for `gh issue comment 53 --body-file`.

Phase 3 has two paths: the **short-form comment path** (when Phase 1 fired a smoking gun, §4.3.1) and the **full comment path** (normal flow, §4.3.2). The orchestrator chooses based on Phase 1's smoking-gun verdict; if the scope is `both`, only the short-form path runs. If the scope is one use case only, the comment is *split*: short-form REJECT for the killed use case, full-form content for the surviving one.

#### 4.3.1 Short-form comment path (fail-fast on smoking gun)

When Phase 1 returns a smoking-gun verdict of `yes` (scope `both`, `headless-CI`, or `agentic-dev`), Phase 2 is skipped for the killed scope and the synthesis collapses to a short-form comment. The synthesis subagent is still dispatched if the orchestrator wants the comment written with the standard tone and footer; if it is overkill, the orchestrator may write the short-form comment directly.

Short-form comment structure:

1. **Preamble** — one short paragraph restating the pessimistic-prior framing from §2 and naming the use-case scope of the rejection. End with a one-line recommendation tag: `**Recommendation: REJECT (use case: <scope>)**`.
2. **Smoking-gun finding** — one short paragraph, ≤4 sentences, naming the disqualifier that uniformly killed all four candidates (or the structural GCP-for-this-use-case blocker), with the source URLs from Phase 1.
3. **Degenerate comparison table (optional).** A single-column-of-content table with rows for all four candidates and a shared "killed by smoking gun: <reason>" cell, or omit the table entirely if it adds no information. The synthesis subagent picks whichever is clearer.
4. **One-line REJECT recommendation** for the killed use case, with a "what would change our mind" trigger (same shape as the full-comment DEFER/REJECT clause — at least one observable / quantitative trigger).
5. **Price-drift caveat** — same one-sentence reminder as the full comment.
6. **Cross-references footer** — same `#48, #44, #9, #35, #20` block as the full comment.

**Comment length target for short-form:** roughly 80–200 lines of markdown.

The PR description, in the short-form case, summarizes the smoking gun in one line (e.g., "REJECT — uniform paid-account-plus-quota requirement across all four candidates makes GCP non-viable as a headless CI target as of 2026-05-20") and links to the issue comment.

#### 4.3.2 Full comment path (normal flow)

**Required structure of the comment:**

1. **Preamble** — one short paragraph restating the pessimistic-prior framing from §2 (so a future reader understands the bar). End with a one-line recommendation tag: `**Recommendation: ADOPT | DEFER | REJECT**` — or, where the two use cases warrant different verdicts, a **split recommendation** of the form `**Recommendation: <verdict-CI> for headless CI; <verdict-dev> for agentic dev**`.
2. **Candidates considered.** Bulleted list of all four, with a one-line "obviously-wrong-fit" note for any candidate the triage killed on a single disqualifier (per the issue's deliverable section 1).
3. **Comparison tables.** Two tables — one for the **headless CI** use case, one for the **agentic dev** use case. Each table has one row per candidate, including killed candidates.

    **Table A — Headless CI fit.** Columns:

    | Column | Contents |
    | --- | --- |
    | Candidate | e.g., "Vertex AI Custom Jobs". |
    | Cost/PR (best — spot) | $X.XX at the assumed 5-min job length on the cheapest SKU the candidate offers (typically T4 spot). `n/a` if killed. |
    | Cost/PR (worst — on-demand) | $X.XX at the assumed 10-min job length on A100 on-demand. `n/a` if killed. |
    | Cold-start | "<N min" sourced. `n/a` if killed. |
    | GPU pinnability | "T4 / L4 / A100 selectable per job" or "T4-only" or similar. `n/a` if killed. |
    | Headless-ness | "WIF" / "SA key only" / "interactive only". |
    | HF gated checkpoint friction | "low (secret env var + pip)" / "medium (Secret Manager + IAM binding)" / "high (no outbound egress)". |
    | Kill reason | Empty for survivors; the disqualifier(s) for killed candidates. |

    **Table B — Agentic dev fit.** Columns:

    | Column | Contents |
    | --- | --- |
    | Candidate | e.g., "Vertex AI Custom Jobs". |
    | Interactive surface | "SSH" / "JupyterLab terminal" / "web-shell" / "none (request-response only)". `n/a` if killed at triage. |
    | Persistent workspace | "boot disk + persistent disk" / "GCS mount" / "stateless (re-clone)". `n/a` if killed. |
    | Claude Max auth | "browser OAuth feasible" / "device-code only" / "structurally infeasible". `n/a` if killed. |
    | $/2h-session (best — spot) | $X.XX at the cheapest GPU SKU the candidate offers, 2 hours. `n/a` if killed. |
    | $/2h-session (worst — on-demand) | $X.XX at the cheapest GPU SKU the candidate offers, 2 hours, on-demand. `n/a` if killed. |
    | Kill reason | Empty for survivors; the disqualifier(s) for killed candidates. |

4. **Cost-vs-value lens** (the load-bearing section — see §5). Two short paragraphs — one per use case — explicitly computing the trade-off. Headless CI form: "GCP costs $X-Y/PR but eliminates Z minutes of human-re-run-Colab toil per PR." Agentic dev form: "GCP costs $X-Y per 2-hour session for L4/A100-class GPU access, against a Colab Free baseline of $0/session but T4-only and time-capped."
5. **Recommendation block.** Addresses **both** use cases. May be a single verdict (if the candidate set scores the same way on both) or a split verdict (e.g., "REJECT for headless CI, DEFER for agentic dev — pending free-trial GPU access").
    - **If ADOPT (for either or both use cases):** A "smallest-viable-PR sketch" for each adopted use case. For headless CI: bullet list of (a) one Vertex Custom Job YAML (or equivalent for the recommended candidate), (b) the WIF setup steps (OIDC provider + IAM binding), (c) a draft `.github/workflows/gpu-tests.yml` skeleton. For agentic dev: bullet list of (a) instance-creation sketch (SKU, persistent-disk size, region), (b) Claude Code install + login sequence inside the instance, (c) workspace-persistence sketch. These are *sketches in the comment body* — fenced code blocks for illustration — not committed files.
    - **If DEFER (for either or both use cases):** A "what would change our mind" clause per use case. Examples for headless CI: *"if GCP introduces free-trial GPU access in `us-central1`"*; *"if a new GCP SKU brings T4-spot cost-per-PR under $0.05"*. Examples for agentic dev: *"if Colab Pro+ ceases to provide reliable L4 access"*; *"if a new GCP SKU brings spot L4 under $X/hour"*. At least one trigger per use case must be quantitative or otherwise observable so the deferral has a concrete reopen condition.
    - **If REJECT (for either or both use cases):** Same as DEFER but with a stronger framing — re-evaluation is not expected absent a structural change at GCP, not just a price drift.
6. **Price-drift caveat.** A one-sentence reminder that all dollar figures are as of 2026-05-20 and require re-validation if the issue is revisited later.
7. **Cross-references.** A "Related issues" footer linking #48, #44, #9, #35, #20 with the one-line "why related" notes from §6 below.

**Comment length target:** roughly 400–800 lines of markdown. Long enough to carry the two tables + the per-candidate rationales (across both use cases) without burying the recommendation; short enough that the recommendation is visible without scrolling past three screens.

---

## 5. Cost-vs-Value Framing (load-bearing in the synthesis)

The cost-vs-value lens runs **twice — once per use case** — in the synthesis comment.

### 5.1 Headless CI use case

The recommendation must explicitly compute three numbers:

1. **GCP cost per PR.** Best-case spot $/PR and worst-case on-demand $/PR for the *recommended* candidate (or for the cheapest survivor if the recommendation is DEFER/REJECT). Sourced figures, "as of 2026-05-20".
2. **Setup tax.** One-time, in hours. Covers: WIF OIDC provider creation, IAM binding for the project's service account, quota-increase request for the chosen GPU SKU, first-job smoke. The synthesis subagent estimates this from Phase 2's auth-path brief.
3. **Per-PR human-in-the-loop tax of the current Colab flow.** Minutes per PR spent on the "human re-runs Colab on flakes" loop. Source: the owner's own anecdotal estimate. The synthesis comment should ask explicitly *"is this estimate right?"* in a footnote — the owner reading the comment is the only ground truth.

The recommendation block then frames the trade-off in plain language:

> "GCP costs $X-Y/PR but eliminates Z minutes/PR of human-re-run-Colab toil, against a one-time setup tax of T hours. At the current PR cadence of ~N PRs/month, breakeven on the setup tax is M months."

### 5.2 Agentic dev use case

The recommendation must explicitly compute three parallel numbers:

1. **GCP cost per 2-hour agentic session.** Best-case spot $/session and worst-case on-demand $/session for the *recommended* candidate on its cheapest GPU SKU (rough proxy for an agentic-dev session). Sourced figures, "as of 2026-05-20".
2. **Setup tax.** One-time, in hours. Covers: instance creation and persistent-disk attach, base image / toolchain install (CUDA, Python env, repo clone), Claude Code CLI install and Claude Max OAuth login from inside the instance, first-session smoke. Mostly overlaps with the headless-CI setup tax, but adds Claude Code install + login.
3. **Per-session value.** The capability the candidate unlocks that the Colab Free baseline does not: ability to use higher-tier GPUs (L4, A100) for codepaths that the Colab T4 cannot exercise (#9), minus the value the Colab Free baseline already delivers (real but T4-only and time-capped). The synthesis comment expresses this as a qualitative gap, not a dollar figure — the value side of the agentic-dev trade-off is capability, not cost-saved.

The recommendation block then frames the trade-off in plain language:

> "GCP costs $X-Y per 2-hour session and unlocks L4/A100-class codepaths the Colab T4 cannot exercise (#9), against a setup tax of T hours (including Claude Code install + Claude Max login). The Colab Free baseline remains $0/session but T4-only and time-capped."

### 5.3 Synthesis tone

Both framings must be **explicit**, not implicit. A reader scanning the comment for "is this worth it?" should find the trade-off computed in one paragraph per use case, not buried across the brief. The split-recommendation case (different verdict per use case) requires both paragraphs side-by-side so the asymmetry is legible.

---

## 6. Cross-References to Related Issues

The synthesis comment's footer lists these with one-line notes; the planner / synthesis subagent does **not** need to re-derive these — they are fixed here:

- **#48** — current Colab-based GPU test loop and the per-branch PR badge. The "dev-loop fit" axis (§4.2 axis 5) is anchored against this loop; the baseline being undercut is #48's $0/PR cost.
- **#44** — manual GPU test pass cadence. Establishes that GPU runs are not free-as-in-effort today; the per-PR human-in-the-loop tax (§5 number 3) is exactly this cadence's cost.
- **#9** — sm_75+ codepaths that need real-hardware coverage. Motivates the "complements" verdict option in §4.2 axis 5 (headless CI: a candidate that pins L4/A100 could exercise codepaths the Colab T4 cannot) **and** the per-session value side of the agentic-dev cost-vs-value framing in §5.2 (agentic dev: an L4/A100 instance lets the owner iterate interactively on those same codepaths). Both use cases benefit from L4/A100 access.
- **#35** — parallel investigation for AWS / Lambda Labs as security-conscious targets. Explicitly out of scope here; the comparison table does **not** include AWS/Lambda rows, only the four GCP candidates.
- **#20** — earlier sketch for AWS SageMaker Studio Lab as a test harness; same shape (cloud headless GPU CI), different cloud. Out of scope here; mentioned only to make clear this evaluation does not subsume it.

---

## 7. Pricing-Snapshot Policy

Concrete dollar numbers are mandatory (the issue explicitly asks for cost-per-PR). To prevent rot:

- Every dollar figure in any Phase 2 brief or in the Phase 3 comment carries an explicit **"as of 2026-05-20"** annotation.
- Every dollar figure carries a source URL — the GCP pricing page for the SKU, the Cloud Billing Calculator output URL, or a credible community post (in that order of preference).
- The synthesis comment's price-drift caveat (§4.3 item 6) explicitly says prices drift and that any future reader revisiting the recommendation must re-validate.
- Spot vs. on-demand figures are reported as a pair, never just one of the two; spot is the lower bound, on-demand is the upper bound, and the recommendation framework (§4.3 item 5) is allowed to lean on either depending on whether the recommended candidate supports spot reliably for short jobs.

---

## 8. Acceptance Criteria

- [ ] Phase 1 produces a kill-or-survive verdict for all four candidates with sourced rationale; no "TBD" entries.
- [ ] Phase 1's return value includes an explicit **smoking-gun verdict line** (`yes` | `no`, with scope and rationale if `yes`) per §4.1.1.
- [ ] **Branching:** if Phase 1's smoking-gun verdict is `yes` with scope `both`, the deliverable is the **short-form comment** described in §4.3.1 (preamble + smoking-gun finding + optional degenerate table + one-line REJECT with trigger + price-drift caveat + cross-references footer), and the acceptance criteria below that reference Phase 2 / full-comment Phase 3 content do **not** apply. If the scope is one use case only, the comment is split: short-form REJECT for the killed use case, full-flow content for the surviving one (and the criteria below apply only to the surviving use case's content).
- [ ] **Normal flow:** Phase 2 produces one deep-dive brief per surviving candidate covering all **seven** axes from §4.2 (axes 1–6 for headless CI, axis 7 for agentic dev); each brief is parallel-dispatchable (no cross-candidate dependencies). For a use-case-specific smoking gun, only the surviving use case's axes are required.
- [ ] **Axis 7 coverage:** every surviving candidate's Phase 2 brief covers all five sub-points of axis 7 (interactive surface, persistent workspace, Claude Max auth path, agentic-dev cost dynamics, Claude Max usage-pattern posture), unless the agentic-dev use case was killed by a use-case-specific smoking gun.
- [ ] **Normal flow:** Phase 3 produces a markdown file ready to post as a comment on issue #53. The comment contains all seven sub-sections from §4.3.2, including **both** comparison tables (Table A — Headless CI fit, Table B — Agentic dev fit) with rows for **every** candidate (including killed ones).
- [ ] **Normal flow:** the comment contains the **explicit cost-vs-value trade-off paragraphs** (§5) — one per use case, with the three numbers per use case computed in plain prose.
- [ ] **Normal flow:** the recommendation block addresses **both** use cases (headless CI and agentic dev). It may be a single shared verdict or a split verdict (e.g., REJECT for headless CI, DEFER for agentic dev).
- [ ] The recommendation is one of ADOPT / DEFER / REJECT (or a split of two such verdicts per use case), in the comment's preamble and again in the recommendation block.
- [ ] If ADOPT (for either or both use cases): the smallest-viable-PR sketch for each adopted use case is present in the comment body as fenced code blocks. No file under `.github/workflows/` is added to the repo.
- [ ] If DEFER or REJECT (for either or both use cases): at least one observable / quantitative trigger is named in the "what would change our mind" clause for each deferred / rejected use case.
- [ ] Every dollar figure carries "as of 2026-05-20" + source URL.
- [ ] Every candidate-specific quantitative claim (cold-start, quota timeline, $/2h-session, interactive-surface availability, Claude Max auth feasibility) carries a source URL.
- [ ] Cross-references #48, #44, #9, #35, #20 each appear in the comment with the one-line "why related" notes from §6 (including the updated #9 note covering both use cases).
- [ ] The PR landing this work contains only the spec and the implementation plan under `docs/superpowers/{specs,plans}/`; no files under `.github/`, no files under `docs/` outside the superpowers tree.
- [ ] Issue #53 closes when the PR merges (via the comment's posting and a "Closes #53" line in the PR description). The PR description's one-line recommendation reflects the short-form-vs-full-flow path Phase 3 took.

---

## 9. Out of Scope (Deferred, Tracked Elsewhere)

- **Hands-on GCP setup, account conversion, quota requests, smoke tests.** Strictly investigation-only per #53.
- **Live billing data.** The evaluation uses published pricing + Cloud Billing Calculator estimates; no real-account spend.
- **AWS / Azure / Lambda Labs.** Tracked under #35.
- **SageMaker Studio Lab.** Tracked under #20.
- **Workflow YAML actually committed to the repo.** The smallest-viable-PR sketch in the synthesis comment is illustrative; the implementation PR is a separate future PR, gated on the recommendation being ADOPT.
- **Replacing the user-facing Colab beginner notebook** `notebooks/esam3_train.ipynb` — out of scope per the issue.
- **Re-evaluating the GPU test policy or its marker tiers.** The evaluation assumes `2026-05-19-gpu-test-policy-design.md` as fixed.
- **Comparing GCP against itself across multiple regions.** The deep-dives pick one primary region per candidate (typically `us-central1` for SKU breadth) and call it out; multi-region cost arbitrage is not investigated.
- **Setting up Anthropic API keys for non-Claude-Max usage** (e.g., a service-account API key for headless agent runs, billed via the Anthropic Console). The agentic-dev axis (§4.2 axis 7) assumes the owner's existing **Claude Max OAuth login** is what runs Claude Code from inside the instance, not a separate API key. Evaluating the API-key path is a separate question and is not investigated here.

---

## 10. Open Questions

These items genuinely could not be resolved while writing the spec. The planner subagent should resolve them inline (preferred) or escalate to the user per the design-ambiguity ladder.

1. **PR-run length assumption — 5 min, 10 min, or both endpoints?** Phase 2 axis 3 currently asks for "the resulting per-PR cost at both endpoints of the 5–10 minute window," which doubles the number of dollar figures per candidate. Acceptable, but the synthesis table (§4.3 item 3) only has columns for "best-case spot" and "worst-case on-demand" — collapsing the 5/10-minute axis into "best/worst" hides the duration assumption. The planner should pick one of: (a) report both endpoints in the brief and average to one number for the table; (b) report only the 10-min worst-case in the table (conservative); (c) add a duration column to the table. Recommendation: option (b) for table simplicity, with both endpoints retained in the deep-dive briefs.
2. **Can triage early-kill a candidate before checking WIF?** The disqualifier checklist is presented as four parallel checks, but the cheapest disqualifier (paid-account-required) might fire first and make the WIF/HF checks moot. The planner should decide whether the triage subagent is allowed to skip later checks once any check fires (saves research time) or must complete all four for every candidate (more thorough; gives a stronger comparison-table cell). Recommendation: allow early-kill, but require the verdict block to state which checks were not evaluated and why.
3. **Where does the synthesis subagent get the per-PR human-in-the-loop tax number?** §5 says "the owner's own anecdotal estimate." The planner should decide whether to (a) ask the user for the number as a one-shot question before Phase 3 dispatches, (b) have the synthesis subagent leave a `<NEEDS OWNER ESTIMATE>` placeholder and the orchestrator fills it in before posting, or (c) make the synthesis subagent estimate it from the #44 issue body. Recommendation: option (a) — single clarifying question to the user before Phase 3.
4. **Should Cloud Run GPU be in the candidate list at all?** Cloud Run GPU is the newest of the four and the most likely to be killed at triage (Cloud Run is fundamentally an HTTP-request runtime, not a batch-job runtime; the impedance mismatch with `pytest -m gpu` is large). The issue explicitly names it as a candidate, so it stays in. The planner does not need to revisit this — noted here so the triage subagent does not relitigate the inclusion decision.
5. **PR description vs. synthesis comment — overlap?** §3 says the PR description "summarizes the recommendation (one line: adopt / defer / reject) and links to the issue comment." The planner should decide whether the PR description should also paste the comparison table (more redundant but more discoverable from the PR view) or remain a one-liner (cleaner, but a reader of the PR has to click through). Recommendation: one-liner in the PR description with a link, table only in the issue comment. In a split-recommendation case, the one-liner names both verdicts (e.g., "REJECT for headless CI, DEFER for agentic dev").
6. **Agentic-dev session-length assumption — single value or range?** §4.2 axis 7 sub-point (d) and §5.2 number 1 anchor on a 2-hour-per-day session as a rough proxy for an agentic-dev session. The planner should decide whether the brief reports a single primary length (cleaner table, single $ number per cost cell) or a range (e.g., 1h / 2h / 4h, more informative but multiplies the dollar figures). Recommendation: single primary length (2 hours) with a one-sentence note that the user can scale the figure linearly to other session lengths.
7. **Claude Code login flow — browser OAuth vs. device code?** The Claude Code CLI's standard login is browser-based OAuth, which assumes a usable browser on the same machine. From inside a headless cloud VM, the planner should decide whether the agentic-dev brief is required to confirm a working login path (device-code flow, SSH port-forward for the OAuth redirect, or copy-paste of a token from a local login) or whether stating "browser OAuth is the documented path; the owner will verify on first use" is acceptable. Recommendation: the brief states the documented path per the Claude Code docs and flags non-trivial workarounds; live verification is out of scope (consistent with §1.3's hands-on-setup exclusion).
