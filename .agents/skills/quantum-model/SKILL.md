---
name: quantum-model
user-invocable: false
description: Use when a user names or describes a quantum model covered by the project and needs its model card, conventions, observables, or method guidance.
---

# model dispatcher

Auto-triggered. The user does not type `/model`; the description above fires
the skill when their prose names a harness-tracked model.

## Audience definition (binding)

<audience>
The reader is a working physicist with no harness-internal context. They
want the result with embedded reasoning (what method, why, what was
verified), not the agent's process. They do NOT know harness vocabulary
(manifest, deviation). Every user-facing line is anchored to this audience.
</audience>

## Workflow

1. **Match.** Resolve user's prose to one canonical model name. Handle
   aliases (TFIM → transverse-field-ising, SIAM → anderson-impurity, …).
   Consult [model names and aliases](references/model-index.md) when needed.
   Resolve each relevant model when the task involves more than one.
2. **Read the card.** Use `.knowledge/models/<name>/MODEL.md` for project
   conventions on first use. Reuse it while its relevant content remains in
   context and unchanged; reread when context is missing, the file changes, or
   conflicting evidence appears. Check current or contested claims against
   primary sources when needed. Before computing, establish:

   <checklist name="card-read">
   - Hamiltonian definition and sign/normalization conventions read
   - Declared phases and their order parameters identified
   - Observables and their canonical forms noted
   - Recommended method(s) and their stack noted
   - Verification rubric (limit / symmetry / convergence / cross-method) noted
   </checklist>

3. **Serve.** Surface the card's facts relevant to the moment — Hamiltonian
   and conventions, phases and observables, method recommendations,
   verification pointers — into whatever workflow is active
   (`/reproduce-paper`, `/solve`, a method skill). The card informs that
   workflow; it does not re-route or replace it.

## Source use

Do not substitute generic defaults for the card's project conventions. Cite the
card supporting the result. Local cards can become outdated; resolve conflicts
with the user's stated setup and relevant primary sources rather than treating
any card as infallible.
