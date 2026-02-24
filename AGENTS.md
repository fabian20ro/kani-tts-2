# AGENTS.md

> This file provides non-discoverable bootstrap context.
> If the model can find it in the codebase, it does not belong here.
> For corrections and patterns, see LESSONS_LEARNED.md.

## Constraints

- Dev deps are optional groups: `pip install -e ".[dev]"` (pytest, black, isort, flake8), `.[mlx]` (Apple Silicon), `.[audio]` (soundfile)
- Requires `transformers==4.56.0` exactly — the model imports from `transformers.models.lfm2` which is version-specific
- PyTorch and MLX are parallel backend implementations — MLX is lazily imported in `__init__.py`, never add it as a hard dependency
- Changes to token layout (`_tokens.py`) or position encoding logic MUST be mirrored in both backends
- Shared modules (`_tokens.py`, `_utils.py`) must NOT import `torch` or `mlx` directly
- Tests never load real models — all model logic is tested via mock objects or manual token computation; each test file is self-contained (no conftest.py or shared fixtures)

## Legacy & Deprecated

<!-- Nothing currently deprecated. Add entries here when patterns are superseded. -->

## Learning System

This project uses a persistent learning system. Follow this workflow every session:

1. **Start of task:** Read `LESSONS_LEARNED.md` — it contains validated corrections and patterns
2. **During work:** Note any surprises or non-obvious discoveries
3. **End of iteration:** Append to `ITERATION_LOG.md` with what happened
4. **If insight is reusable and validated:** Also add to `LESSONS_LEARNED.md`
5. **If same issue appears 2+ times in log:** Promote to `LESSONS_LEARNED.md`
6. **If something surprised you:** Flag it to the developer — they'll decide whether to fix the codebase, update LESSONS_LEARNED, or adjust this file

| File | Purpose | When to Write |
|------|---------|---------------|
| `LESSONS_LEARNED.md` | Curated, validated wisdom and corrections | When insight is reusable |
| `ITERATION_LOG.md` | Raw session journal (append-only, never delete) | Every iteration (always) |

Rules: Never delete from ITERATION_LOG. Obsolete lessons → Archive section in LESSONS_LEARNED (not deleted). Date-stamp everything YYYY-MM-DD. When in doubt: log it.

### Periodic Maintenance
This project's config files are audited periodically using `SETUP_AI_AGENT_CONFIG.md`.
The maintenance protocol ensures all files stay lean and current. See that document's
"Periodic Maintenance Protocol" section for the full audit procedure.

## Sub-Agents

Specialized agents in `.claude/agents/`. Invoke proactively — don't wait to be asked.

| Agent | File | Invoke When |
|-------|------|-------------|
| Architect | `.claude/agents/architect.md` | System design, module boundaries, cross-backend parity |
| Planner | `.claude/agents/planner.md` | Complex multi-step features — plan before coding |
| Agent Creator | `.claude/agents/agent-creator.md` | Need a new specialized agent for a recurring task domain |
