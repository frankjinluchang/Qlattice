# `qlat.cgfhmc` — Canonical Gauge-Fixed HMC (CGFHMC)

Source: `qlat/qlat/cgfhmc.py`

> **Note:** Update this document when updating the source file.

## Outline

1. [Overview](#overview)
2. [The Algorithm](#the-algorithm)
   - [Gauge Fixing Inside the Trajectory](#gauge-fixing-inside-the-trajectory)
   - [Partial Momentum Refresh](#partial-momentum-refresh)
   - [Metropolis on the Corrected Energy](#metropolis-on-the-corrected-energy)
3. [Caveats](#caveats)
4. [The `c(k)` Field](#the-ck-field)
5. [API Reference](#api-reference)
   - [Momentum Grid](#momentum-grid)
   - [Building a `c(k)` Field](#building-a-ck-field)
   - [The Refresh](#the-refresh)
   - [The Driver](#the-driver)
6. [Example](#example)
   - [Choosing `md_time` and `n_step`](#choosing-md_time-and-n_step)
   - [What the Example Records](#what-the-example-records)

---

## Overview

`cgfhmc` implements **Canonical Gauge-Fixed HMC (CGFHMC)** for pure SU(3)
lattice gauge theory. It is the canonical-form HMC of a gauge-fixed lattice:
a standard force-gradient HMC trajectory with gauge fixing **inside** the
molecular-dynamics evolution, plus a Fourier mode-dependent partial momentum
refresh.

The module is pure Python. Every primitive it uses is already exposed by
`qlat.hmc`, `qlat.hmc_utils`, `qlat.gauge_fixing` and `qlat.field_utils_utils`,
so it adds no compiled code.

In one line:

> CGFHMC = canonical HMC + Landau gauge fixing after every MD step (applied to
> both `gf` and `gm`) + partial momentum refresh, with Metropolis on the energy
> violation corrected by the refresh-induced kinetic change.

## The Algorithm

One trajectory is a standard force-gradient HMC trajectory (the same uniform
symmetric 4th-order Omelyan integrator as
`qlat.hmc_utils.run_hmc_evolve_pure_gauge`) with two insertions after every
**complete** MD step, at the same symmetric phase point.

### Gauge Fixing Inside the Trajectory

After each complete MD step the gauge field is Landau-gauge-fixed and the
**same** transform is applied to the gauge field and to the gauge momentum:

```python
(gt, info) = q.fix_landau_gauge(gf, max_iter, eps, alpha)
gf @= gt * gf        # field
gm @= gt * gm        # momentum (same transform, adjoint action)
```

Applying the transform to `gm` as well is what makes the algorithm
*canonical*: the momentum is rotated with the field rather than left in the old
gauge, so `(gf, gm)` stays a canonical pair and the kinetic energy is invariant
under the fixing. `qlat.gauge_fixing.fix_landau_gauge` does not mutate `gf`; it
returns `(gt, info)` with `gt * gf` the fixed field.

The momentum transform is the adjoint action at the **same site**, with no
`x + mu` shift — `gf_evolve` left-multiplies, so gauge covariance requires it.
See `q.gm_apply_gauge_transformation` and the `gt * gm` sugar.

### Partial Momentum Refresh

After the fixing, the momentum is partially refreshed with a mode-dependent
Fourier coefficient `c(k)`:

$$
P_k \leftarrow \sqrt{1 - c(k)^2}\, P_k + c(k)\, \xi_k
\tag{1}
$$

where `xi` is a fresh Gaussian gauge momentum. Here `c` is the **fresh-noise
weight**, not the correlation:

- `c = 0` — strict no-op: draws no random number and leaves the momentum
  bit-identical;
- `c = 1` — complete resampling of that mode;
- the retained-momentum autocorrelation is `sqrt(1 - c(k)^2)`.

The older convention stored the correlation `c_corr`; the conversion is
`c = sqrt(1 - c_corr^2)` (so `c_corr = 0.9` corresponds to `c = 0.4359`).

The refresh preserves the Gaussian momentum ensemble **mode by mode**. With
`E|P_k|^2 = sigma_m^2`,

$$
E|P'_k|^2 = (1 - c(k)^2)\, \sigma_m^2 + c(k)^2\, \sigma_m^2 = \sigma_m^2
\tag{2}
$$

so `c(k)` sets only the mode-dependent decorrelation rate. Both `P_k` and
`xi_k` carry qlat's native per-component variance, so no rescaling is needed.
A constant `c` reduces to the usual Ornstein-Uhlenbeck refresh.

A **constant** `c` is applied directly in real space: the unitary Fourier
transform commutes with a scalar mixing, so the result is mathematically
identical and much cheaper. This is the fast path; `info["path"]` reports which
route ran.

### Metropolis on the Corrected Energy

The stochastic kinetic change injected by the refresh is removed from the
Metropolis variable:

- `delta_h = H_final - H_initial` — the raw trajectory value, which is what the
  driver returns;
- `dK_refresh_sum = sum over steps of (K_after - K_before)`;
- `delta_h_accept = delta_h - dK_refresh_sum` — the value passed to
  `q.metropolis_accept`.

Returning the raw `delta_h` keeps the meaning identical to
`run_hmc_pure_gauge`, so the two drivers' values stay comparable.

## Caveats

**Exactness.** For a **uniform** `c` this is the standard exact
Ornstein-Uhlenbeck partial momentum refresh and the algorithm is exact HMC. For
a **mode-dependent** `c(k)` the refresh is a global Fourier-space operation and
is not a standard partial refresh, so removing `dK_refresh_sum` from the
Metropolis variable is an adopted convention supported by empirical sampling
checks, not a formal generalized-HMC proof. Do not present mode-dependent
CGFHMC as an exact HMC.

**Reversibility.** CGFHMC is **not time-reversible**: `fix_landau_gauge` is a
projection onto one representative per gauge orbit, which discards the
phase-space information a reverse test relies on. Measured on a `2^4` lattice
with 8 MD steps, forward then `md_time = -1`, the reverse residual is of order
`1e-15` without the per-step fixing and of order `1e-1` with it.

`run_hmc_cgfhmc_pure_gauge` therefore accepts `is_reverse_test` for signature
compatibility with `qlat.hmc_utils.run_hmc_pure_gauge`, but raises
`NotImplementedError` when it is `True` rather than returning a meaningless
verdict.

## The `c(k)` Field

`c(k)` is a plain 4-D `float64` numpy array shaped like the lattice in
`total_site` axis order `(Lx, Ly, Lz, Lt)` — the order
`q.Geometry(...).total_site` uses, not the t-first order used for qlat-ref link
arrays.

It must satisfy `0 <= c <= 1` and be even under the 4D central inversion
`c(-k) = c(k)`. Both are required: `sqrt(1 - c^2)` must be real, and the su(3)
component fields are real so their spectra obey `X_{-k} = conj(X_k)` — only
even coefficients keep the refreshed momentum real and anti-Hermitian.
`regularize_c_field` enforces both (it clips and averages with the
central-inversion partner), and every field-building helper applies it.

The field is small (2 KiB at `4^4`, 131 KiB at `16^3 x 32`) and must be
identical on every node; it carries no profile logic, so any prescription —
analytic, tabulated or loaded from a file — enters through the same path.

## API Reference

### Momentum Grid

- **`k_vectors(geo)`** → `(4, Lx, Ly, Lz, Lt)` `float64`. Component `i` is the
  physical momentum along lattice axis `i`, varying along its own axis, with
  the folded (signed) index covering `[-pi, pi)` in FFT convention:
  `k_i[n] = (2*pi/L_i) * n` for `n <= L_i/2`, else `(2*pi/L_i) * (n - L_i)`.
- **`k_squared(geo)`** → `(Lx, Ly, Lz, Lt)`, equal to
  `np.sum(k_vectors(geo)**2, axis=0)`.
- **`resolve_k_cut(geo, units)`** → `float`, `units * 2*pi/Lx`. A convenience
  for writing prescriptions: `resolve_k_cut(geo, 2)` on `Lx = 16` is
  `2*pi/8`.

`geo` must be a `qlat.Geometry`; a tuple or bare shape is rejected with
`TypeError` so the mode grid is guaranteed to match the gauge field.

### Building a `c(k)` Field

- **`generate_c_field(geo, func)`** — calls `func(k_vec)` once with the
  `(4, Lx, Ly, Lz, Lt)` momentum array and returns a regularized, C-contiguous
  `float64` field. Regularization is always applied; the output must be real
  and broadcastable to the lattice shape.
- **`regularize_c_field(c, *, in_place=False, allow_nan="raise")`** — clip to
  `[0, 1]` and enforce `c(-k) = c(k)`. `allow_nan` is `"raise"` (default) or
  `"zero"` (map non-finite entries to `0.0`, a full refresh of those modes).
  Diagnostics go to `regularize_c_field.last_report`.
- **`c_field_from_file(path)`** — load a `.npy` field and regularize it.

### The Refresh

**`partial_refresh_momentum_mode(gm, rng, c_field)`** → `(applied, dK, info)`.

Applies the refresh to `gm` **in place**. `applied` is `False` iff `c(k) == 0`
everywhere, in which case no random number is drawn and `gm` is bit-identical,
so runs with the refresh disabled match runs without it exactly. `dK` is
`K_after - K_before`, MPI-reduced; the caller excludes it from the Metropolis
variable. `info` holds `c_min`/`c_max`/`c_mean`, `n_modes`,
`n_modes_refreshed`, `n_modes_frozen`, `is_uniform`, `no_op` and `path`.

The Fourier route uses qlat's **distributed** FFT with `is_normalizing=True`
(the unitary convention, `norm="ortho"` in numpy terms) and works on any node
decomposition; after the transform the momentum-space modes are indexed through
`geo.xg_arr()`. The real-space fast path handles a constant `c`.

### The Driver

```python
run_hmc_cgfhmc_pure_gauge(
    gf, ga, traj, rs,
    *,
    c_field,
    is_reverse_test=False,
    n_step=6,
    md_time=1.0,
    is_always_accept=False,
    gauge_fix_max_iter=25,
    gauge_fix_eps=1.0e-10,
    gauge_fix_alpha=0.1,
    info_out=None,
) -> (flag, delta_h)
```

The positional and shared keyword-only parameters mirror
`qlat.hmc_utils.run_hmc_pure_gauge` exactly, in the same order with the same
defaults; the CGFHMC-specific parameters follow.

- `gf` is updated in place on acceptance (or always with
  `is_always_accept=True`).
- `delta_h` is the **raw** trajectory value `H_final - H_initial`.
- `gauge_fix_max_iter=25` is deliberately much smaller than
  `fix_landau_gauge`'s own default of 10000: the field is re-fixed every MD
  step and starts already near Landau gauge, so only the first step after a
  cold start needs many sweeps.
- `info_out`, when a dict is passed, is filled in place with `dK_sum`,
  `n_refresh_applied`, `n_step`, `n_fix`, `fix_n_iter_total`,
  `fix_n_iter_max`, `fix_n_converged`, `fix_divergence_max`, `delta_h_raw` and
  `delta_h_accept`. The return arity stays 2.
- `is_reverse_test=True` raises `NotImplementedError` (see
  [Caveats](#caveats)).

## Example

The canonical, tested example is
[`examples-py/cgfhmc.py`](https://github.com/jinluchang/Qlattice/blob/master/examples-py/cgfhmc.py);
its test target is `cgfhmc.log`.  It runs two short thermalized chains (a
uniform `c` and a mode-dependent `c(k)`) plus the helper and
refresh-limit checks, in about 7 s under `mpiexec -n 2`.

The essentials, with the parameter values that give a meaningful (mixed, neither
0 % nor 100 %) Metropolis acceptance on a small lattice:

```python
import numpy as np
import qlat as q

size_node_list = [[1, 1, 1, 1], [2, 1, 1, 1]]
q.begin_with_mpi(size_node_list)

total_site = q.Coordinate([4, 4, 4, 4])
geo = q.Geometry(total_site)
beta = 6.0
ga = q.GaugeAction(beta, 0.0)
rs = q.RngState(f"cgfhmc-{total_site.to_list()}")

# A two-zone profile: stiff low modes, hard high modes.  A CONSTANT c would
# take the real-space fast path instead.
k_cut = q.resolve_k_cut(geo, 1)
c_field = q.generate_c_field(
    geo, lambda kv: np.where(np.sum(kv ** 2, axis=0) <= k_cut ** 2, 0.2, 0.436)
)

gf = q.GaugeField(geo)
gf.set_rand(rs.split("init"), 0.3, 2)
gf.unitarize()

# Thermalize with real Metropolis (is_always_accept would drive the action
# uphill and the field would never equilibrate).
traj = 0
for _ in range(8):
    traj += 1
    q.run_hmc_cgfhmc_pure_gauge(
        gf, ga, traj, rs.split("run_hmc_cgfhmc_pure_gauge"),
        c_field=c_field, n_step=3, md_time=1.0,
    )

info = {}
flag, delta_h = q.run_hmc_cgfhmc_pure_gauge(
    gf, ga, traj + 1, rs.split("run_hmc_cgfhmc_pure_gauge"),
    c_field=c_field,
    n_step=3,
    md_time=1.0,
    info_out=info,
)
q.displayln_info(
    f"flag={flag} delta_h={delta_h:.6f} "
    f"dK={info['dK_sum']:.3f} accept={info['delta_h_accept']:.3f} "
    f"fix iterations={info['fix_n_iter_total']}"
)

q.end_with_mpi()
```

### Choosing `md_time` and `n_step`

On a small lattice the raw `delta_h` is dominated by the kinetic-energy
fluctuation: `E[K] = 16 V` and `std(K) = 8 sqrt(V)`, so on `4^4` the kinetic
term alone has a spread of order 64.  A long trajectory with a coarse step size
therefore accumulates a large energy violation and every proposal is rejected.

The example uses `md_time = 1.0` with `n_step = 3` (i.e. `dt = 1/3`), which
keeps `|delta_h_accept|` of order 1–2 and the acceptance mixed (measured
roughly 40 %–50 % depending on the chain and the `c` profile) instead of
always-reject or always-accept.  Note that the raw `|delta_h|` is much larger
(order 45–90), because the refresh injects a kinetic change that the Metropolis
variable removes: what governs acceptance is `|delta_h_accept|`, not
`|delta_h|`.

The step size is deliberately **coarse**.  The 4th-order force-gradient
integrator conserves energy so well that a small `dt` drives the Metropolis
variable itself to machine-noise territory: with the insertion point at the
step boundary, `|delta_h_accept|` scales as `dt^4` (measured on `4^4`,
`beta = 6.0`, `md_time = 0.25`: `3.8e-2` at `dt = 0.125`, `2.4e-3` at
`0.0625`, `1.4e-4` at `0.03125`, `3.6e-7` at `0.0078`).  A recorded
`|delta_h_accept|` of order `1e-5` would be a difference of two `O(50)` numbers,
so a *relative* comparison against a committed reference would be mostly
round-off.  `dt = 1/3` is therefore chosen so that every recorded float stays
well away from zero, matching the "no near-zero quantity with a relative
epsilon" rule.

Two related points from the example's measured behaviour:

- Thermalize with a **real Metropolis** chain.  Using `is_always_accept=True`
  for the thermalization drives the gauge action monotonically uphill, and the
  field never reaches equilibrium (the plaquette stays at its unit-start
  value).  With a random start and Metropolis thermalization the chain
  equilibrates and the acceptance is meaningful.
- The per-step fixing is called with `gauge_fix_max_iter=25`, which is far
  below the `fix_landau_gauge` default of 10000.  On a cold field it will
  report `converged=False` (a fully converged Landau fix needs ~150 sweeps on
  `4^4`); this is expected and matches the validated experimental runs, which
  also used `fix_iter=25` and recorded no converged step.  The divergence still
  drops by several orders of magnitude per step, and the fixing is a
  re-projection of an already nearly-fixed field.

### What the example records

`examples-py/cgfhmc.py` is a log-comparison test: every entry below is compared
against the committed `examples-py/cgfhmc.log.json` by
`q.check_log_json(__file__, check_eps=1e-8)`, and the pass/fail marker is the
single final `CHECK: finished successfully.` line.

- **Floats** (`q.json_results_append(name, value, eps)`): `mean |delta_h|`,
  `mean |delta_h_accept|`, `dK_sum mean abs`, `plaq mean`, and the applied
  `c_min`/`c_max`/`c_mean`.  Absolute values only — a signed mean of `delta_h`
  can land near zero, where a *relative* epsilon is meaningless.
- **c-field / refresh diagnostics**: `c is uniform = ...` names the route the
  refresh actually took (real-space fast path vs Fourier), taken from
  `partial_refresh_momentum_mode`'s own `info` on a throwaway probe momentum, so
  it is the driver's report rather than a re-derivation.  The probe uses its own
  `RngState`, so the chain's random stream is untouched.
- **Counters and flags as strings**, never as fabricated floats: the accept
  count (`accepted = n/10`), `gauge fixations`, `fix steps converged`, whether
  the refresh ran on every MD step, and the `is_reverse_test` guard.
- `OMP_NUM_THREADS=1` / `OMP_DYNAMIC=false` are pinned at the very top, before
  `import qlat`, so the distributed FFT used by the mode-dependent refresh is
  deterministic across runs and machines of the same build.

The example deliberately exercises two chains — a constant `c` (the fast path)
and a two-zone `c(k)` (the Fourier path) — both from a random start with a real
Metropolis warm-up, so `is_always_accept` never hides a broken acceptance.  The
run is deterministic: repeated invocations produce a byte-identical `.log.json`.
