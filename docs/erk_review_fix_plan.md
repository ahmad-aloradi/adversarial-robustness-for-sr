# ERK/ER review — fix plan

Plan to resolve the issues found in the `/code-review max` of the layerwise
sparsity (ERK/ER, RigL arXiv:1911.11134) stack. Scope: the four touched/added
files, docs, plus a one-line typo fix in each of the four Bregman experiment
configs.

## Config typo fix + fail-loud type resolution (fixes #4)

`torch.nn.layernorm.LayerNorm` is a dead entry (`torch.nn.layernorm` does not
import) that `_resolve_layer_types` silently drops today. It appears in all
four Bregman configs — fix to `torch.nn.LayerNorm` in:

- `configs/experiment/sv/sv_bregman_adabreg.yaml:247`
- `configs/experiment/sv/sv_bregman_adabreg_erk.yaml:62`
- `configs/experiment/sv/sv_bregman_linbreg.yaml:240`
- `configs/experiment/sv/sv_bregman_proxsgd.yaml:171`

Then make `_resolve_layer_types` (`pruning_manager.py:30`) fail loud: delete
its `try/except` so an unresolvable entry raises, naming the bad string. This
removes the only silent-drop path and makes a scoped ERK-only diagnostic
unnecessary. The predicate's "specified but none resolved → no match" branch
(`pruning_manager.py:60`) becomes dead and is deleted with R1.

**Resume caveat (accepted):** resuming an existing run reloads the
training-time snapshot from `{exp_dir}/.hydra/`, which still carries the typo
— those resumes now fail loud with the bad string in the message; edit the
snapshot to resume. Eval is unaffected (`PruningManager` is only built in
`configure_optimizers`).

Confirmed safe (verified against shipped configs and tests):

- Every shipped Bregman config defines `lambda_scale` on every group that
  carries a `reg` → #1's raising accessor breaks nothing.
- Every BregmanPruner test that builds optimizer groups sets `lambda_scale`.
- `RegNone` exposes `lamda` (inherits `BregmanRegularizer.__init__`), so
  `_group_has_regularizer` reaches the `lambda_scale` check on norm/bias
  groups, which carry an explicit `lambda_scale: 0.0`.

---

## Two refactors that absorb most of the work

### R1 — remove the per-(param × group) module scan (fixes #2)

`param_matches_config(pl_module, param, config)`
(`src/callbacks/pruning/utils/pruning_manager.py:43`) re-walks
`named_modules()` to locate each parameter's owner, and re-resolves
`layer_types` on **every** call. Replace it with a pure predicate
`module_param_matches(mod_name, mod, p_name, config, resolved_types)` that
takes the owner directly and a pre-resolved type tuple.

- `_process_configs` (`:127`–`:172`): resolve each group's `layer_types` once
  (failing loud per above), iterate
  `named_modules() → named_parameters(recurse=False)` once, assign each param
  to the first matching group (groups stay the inner loop so group-priority
  order is preserved); fallback handling at `:169` unchanged.
- Dedupe tied params by `id` during the walk — `named_parameters()`
  deduplicates by default; the per-module walk must not assign one tied param
  twice.
- `src/callbacks/pruning/utils/layer_shapes.py:33`: call
  `module_param_matches` directly (the loop already has
  `mod_name, mod, p_name`); `_expand_erk_group` resolves the group's types
  once and passes them down — drops the redundant second scan.
- Delete the old owner-searching `param_matches_config`; update the import at
  `layer_shapes.py:11`.

Net: `O(P×G×M)` → `O(P×G)` with cheap checks, and `layer_types` resolved once
per group instead of once per parameter.

### R2 — one predicate, one `lambda_scale` accessor (fixes #1)

- Add `BregmanPruner._lambda_scale(group)` that returns
  `group["lambda_scale"]` and **raises** naming the group when absent (no
  default). Route all six `.get("lambda_scale", …)` sites through it
  (`bregman_pruner.py:546, :560, :569, :580, :732, :808`).
- Rewrite `_setup_layer_schedulers` (`:458`) to build a scheduler for
  **every** `_group_is_erk` group, raising (naming the group) when one lacks
  an active regularizer. The build predicate then equals the consume
  predicate in `_step_layer_schedulers` (`:552`) and `_apply_lambda_to_groups`
  (`:567`), so the `KeyError` on `_layer_schedulers[name]` becomes
  structurally impossible. Keep the existing "regularized non-ERK group"
  guard at `:489`.

---

## Remaining fixes

### #3 — ERK resume fail-loud

`bregman_pruner.py:499` — when `is_resuming and self._erk_mode and
self._ckpt_layer_states is None`, raise (checkpoint lacks
`bregman_erk_layer_scheduler_states`) instead of silently restarting every
per-layer controller from the template defaults.

### #5 — degenerate target (`target_sparsity = 0.0`)

`erk_sparsity.py:137` — delete the `while … else: raise`. The `else` runs on
every no-break exit, so the all-clamped/all-dense solve raises today even
though the docstring admits `target_sparsity=0.0`. After deletion that case
falls through to the existing budget assert (`:147`), which passes for
`target=0.0` (kept == total) and still catches a genuine mismatch.

### #6 — docs contradiction

`docs/pruning.md:143` — reword "converges to the global target **by
construction**…": the per-layer *targets* sum to the global target by
construction; achieved sparsity still depends on each controller reaching its
target, consistent with the Note at `docs/pruning.md:90`.

### #7 — asserts

`pruning_manager.py:_expand_erk_group` — assert the group declares
`layer_types` at the top of the ERK expansion (an ERK group must declare
them). Keep the existing `ndim >= 2` boundary assert at
`layer_shapes.py:35`, and the existing "matched no prunable weights" raise at
`pruning_manager.py:193`. Clear messages, no silent skipping.

### #8 — ramped-ERK `min_epochs`

The non-ERK ramp path bumps `trainer.fit_loop.min_epochs`
(`bregman_pruner.py:261`–`:274`) so the run can't end before the sparsity
ramp finishes; the ERK branch `return`s before that (`:249`). Extract the
bump into a small helper called from both branches — ERK passes any one layer
scheduler (all clones share `warmup_epochs` / `_epochs_to_ramp` /
`_target_initial`). The shipped ERK recipe is fixed-mode
(`_target_initial is None` → helper no-ops), so behavior today is unchanged;
a future ramped ERK template gets the protection for free.

---

## Tests (new behaviors, per repo rules)

- `tests/test_erk_layer_groups.py`: group whose `layer_types` contain an
  unresolvable entry → raises naming the string (typo fix); ERK group missing
  `layer_types` → asserts (#7).
- `tests/test_erk_resume.py`: ERK resume with a checkpoint missing
  `bregman_erk_layer_scheduler_states` → raises (#3); ERK group lacking an
  active regularizer → raises at scheduler setup, and a regularized group
  without `lambda_scale` → raises (#1 / R2 — this file already exercises
  `_setup_layer_schedulers`).
- `tests/test_erk_sparsity.py`: `solve_erk_densities(target_sparsity=0.0)` →
  all densities `1.0`, no raise (#5).
- The existing ERK suites must stay green (R1/R2 are behavior-preserving on
  the shipped path).

---

## Suggested order

Config typo + fail-loud `_resolve_layer_types` → R1 (#2) → R2 (#1) →
#3 → #5 → #7 → #8 → #6 → tests →
`pytest tests/test_erk_* tests/test_bregman_*` and `make format`.
