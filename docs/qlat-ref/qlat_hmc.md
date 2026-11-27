# `qlat_ref.hmc` — Pure-Gauge HMC (NumPy Reference)

Source: `qlat-ref/qlat_ref/hmc.py`, `qlat-ref/qlat_ref/hmc_utils.py`

> **Note:** Update this document when updating the source files.

## Outline

1. [Overview](#overview)
2. [The Module Split](#the-module-split)
3. [Hamiltonian](#hamiltonian)
4. [Momentum Initialisation](#momentum-initialisation)
5. [Force](#force)
6. [Evolution](#evolution)
7. [The Integrator](#the-integrator)
8. [The Trajectory Driver](#the-trajectory-driver)
9. [The RNG Tree](#the-rng-tree)
10. [Mutation Semantics](#mutation-semantics)
11. [Deviation: Single Process](#deviation-single-process)
12. [Examples](#examples)
13. [Self-Checks](#self-checks)

---

## Overview

`qlat_ref.hmc` and `qlat_ref.hmc_utils` implement Hamiltonian Monte Carlo for
a **pure-gauge** lattice, mirroring `qlat/qlat/hmc.pyx` (with its C++ core in
`qlat/qlat/lib/hmc.cpp`) and `qlat/qlat/hmc_utils.py`.

The implementation is a subset of qlat's: the pure-gauge HMC needs no fermions,
no MPI and no communication, so what remains is the part a reader can diff
against the qlat source one function at a time.  Every name, argument order,
default and mutation convention is qlat's.

The trajectory draws a Gaussian momentum, runs the Omelyan force-gradient
integrator for `n_step` steps, and applies the Metropolis test with
probability

$$
P_{\text{acc}} = \min\bigl(1,\ e^{-\Delta H}\bigr)
\tag{1}
$$

where $\Delta H$ is the energy violation of the trajectory.

## The Module Split

| Module | Contents | qlat counterpart |
|---|---|---|
| `qlat_ref.hmc` | `set_rand_gauge_momentum`, `gm_hamilton_node`, `gf_hamilton_node`, `set_gm_force`, `set_gm_force_dual`, `gf_evolve`, `gf_evolve_dual`, `gm_apply_gauge_transformation` | `qlat.hmc` |
| `qlat_ref.hmc_utils` | `metropolis_accept`, `gm_evolve_fg_pure_gauge`, `run_hmc_evolve_pure_gauge`, `run_hmc_pure_gauge` | `qlat.hmc_utils` |

Everything in the table is also exported at the top level of `qlat_ref`, so
`qr.run_hmc_pure_gauge(...)` is the same function object as
`qr.hmc_utils.run_hmc_pure_gauge`.

`gm_apply_gauge_transformation` has **one** implementation, in
`qlat_ref.field` (where the `gt * gm` sugar needs it); `qlat_ref.hmc`
re-exports it as the same object rather than re-wrapping it, so the signature
and behaviour cannot drift apart.

The gauge action value itself lives in `qlat_ref.gauge_action` as
`get_gauge_action(gf, ga)`; `gf_hamilton_node` is the qlat-named entry point
that returns it, so there is one plaquette-plus-rectangle sum in the package.

## Hamiltonian

The energy of a configuration `(U, P)` is `H = K(P) + S(U).`  The kinetic term
is

$$
K = \sum_{x,\mu} \Bigl(-\tfrac{1}{2}\Bigr) \operatorname{Tr}\bigl( P_\mu(x)^2 \bigr)
\tag{2}
$$

computed by `gm_hamilton_node(gm)`, which is the sum of the squares of the
eight Lie-algebra coefficients of every momentum matrix (because the basis
satisfies `Tr[T_a T_b] = -2 delta_ab`).

The gauge part is the Wilson plaquette action with the optional `1 x 2`
rectangle improvement,

$$
S = \frac{\beta}{3} \left[ (1 - 8 c_1) \sum_{P} \bigl(3 - \operatorname{Re} \operatorname{Tr} U_P\bigr) + c_1 \sum_{R} \bigl(3 - \operatorname{Re} \operatorname{Tr} U_R\bigr) \right]
\tag{3}
$$

summed over the `6` oriented plaquettes and `12` oriented rectangles per site,
computed by `gf_hamilton_node(gf, ga)`.  The subtraction of the free-field `3`
per plaquette makes the unit gauge field have `S = 0` exactly, which the
self-checks assert.

## Momentum Initialisation

`set_rand_gauge_momentum(gm, sigma, rng)` fills a `GaugeMomentum` in place with
Gaussian anti-Hermitian traceless matrices distributed as

$$
\varrho(M) \propto \exp\Bigl[ -\operatorname{Tr}\bigl( M^2 \bigr) / \bigl(2 \sigma^2\bigr) \Bigr]
\tag{4}
$$

Note the argument order `(gm, sigma, rng)`: it is qlat's, and it differs from
the method form `gm.set_rand(rng, sigma=1.0)`, which takes the RNG first.  The
function delegates to that method, so there is exactly one implementation of
the draw, and only one place where the RNG layout is defined.

The draw is **deterministic**: for every global site the per-site stream is
`rng.split(str(gindex))` (a string split of the ASCII form of the index, not an
index bump), and each of the four directions draws eight Gaussian
coefficients.

## Force

`set_gm_force(gm_force, gf, ga)` fills the molecular-dynamics force in place:

$$
F_\mu(x) = \operatorname{ah}\Bigl( -\frac{\beta}{3}\, U_\mu(x)\, K_\mu(x)^\dagger \Bigr)
\tag{5}
$$

where $K_\mu(x)$ is the `c1`-weighted staple sum of the link
$U_\mu(x)$ — exactly the sum `gf_hamilton_node` differentiates.  The
projection written `ah` above is the anti-Hermitian traceless part of
`qlat_ref.mat.get_traceless_antihermitian_part`.  The order
`U K^dagger` (not `K^dagger U`) and the `-beta/3` normalisation are pinned by a
finite-difference test, which also asserts that the wrong forms do not pass.

`set_gm_force_dual(gm_force_dual, gf, gm_force)` produces the **dual** momentum
force

$$
F^{\text{dual}}_\mu(x) = -U_\mu(x)^\dagger F_\mu(x) U_\mu(x)
\tag{6}
$$

which the dual evolution `gf_evolve_dual` consumes; the dual momentum lives at
`x + mu` rather than at `x`.

Both functions follow qlat in **initialising their output field themselves**
(`gm_force.init(geo)`), so a momentum created without a geometry, or carrying
another geometry, is accepted and re-initialised.

## Evolution

`gf_evolve(gf, gm, step_size)` updates the links in place,

$$
U_\mu(x) \leftarrow \exp\bigl( h\, P_\mu(x) \bigr)\, U_\mu(x)
\tag{7}
$$

and `gf_evolve_dual(gf, gm_dual, step_size)` right-multiplies the dual
momentum,

$$
U_\mu(x) \leftarrow U_\mu(x)\, \exp\bigl( -h\, P^{\text{dual}}_\mu(x) \bigr)
\tag{8}
$$

In both, the exponential is the truncated Taylor (Horner) series of order `20`
of `qlat_ref.mat.get_su3_exp`, and the product is re-unitarised with the
Gram-Schmidt step of `qlat_ref.mat.get_unitarized` — exactly the arithmetic of
qlat's `matrix_evolve` / `matrix_evolve_dual`.

`gm_apply_gauge_transformation(gm, gt, is_dagger=False)` returns a **new**
momentum with the site-wise adjoint action

$$
P_\mu(x) \leftarrow g(x)\, P_\mu(x)\, g(x)^\dagger
\tag{9}
$$

or its inverse when `is_dagger=True`.  It does not modify its inputs.

## The Integrator

`gm_evolve_fg_pure_gauge(gm, gf_init, ga, fg_dt, dt)` is the force-gradient
momentum half-kick: it evaluates the force at `gf_init`, evolves a **copy** of
the links by `fg_dt` along that force, re-evaluates the force at the
intermediate point, and adds `dt * F_intermediate` to `gm`.  `gf_init` itself
is never mutated.

`run_hmc_evolve_pure_gauge(gm, gf, ga, n_step, md_time=1.0)` is the full
trajectory, mutating **both** `gm` and `gf` in place and returning

$$
\Delta H = H(U_f, P_f) - H(U_i, P_i)
\tag{10}
$$

It is the Omelyan / minimum-norm force-gradient scheme of
`qlat/qlat/hmc_utils.py`, with

$$
\lambda = \frac{1 - 1/\sqrt{3}}{2}
\tag{11}
$$

$$
\theta = \frac{2 - \sqrt{3}}{48}
\tag{12}
$$

and step size `dt = md_time / n_step`; the final field is re-unitarised.
`md_time` may be **negative**, which is the reversibility test: re-running from
the output with `-md_time` returns to the input.

## The Trajectory Driver

`run_hmc_pure_gauge(gf, ga, traj, rs, *, is_reverse_test=False, n_step=6,
md_time=1.0, is_always_accept=False)` runs one trajectory and returns
`(flag, delta_h)`:

1. split the driver seed by the trajectory number, `rs_traj = rs.split(str(traj))`;
2. draw a momentum with `gm.set_rand(rs_traj.split("set_rand_gauge_momentum"), 1.0)`;
3. copy the field (`gf0 = gf.copy()`), run the integrator on the copy, and
   obtain `delta_h`;
4. optionally run the reversed trajectory as a cheap reversibility gate
   (`is_reverse_test`);
5. apply the Metropolis test with `rs_traj.split("metropolis_accept")`;
6. on accept copy the trial field back into `gf` **in place**.

`metropolis_accept(delta_h, traj, rs)` implements step 5: `delta_h <= 0` is
accepted unconditionally with `accept_prob = 1.0`, otherwise
`accept_prob = exp(-delta_h)` and the trajectory is accepted when the uniform
draw `rs.u_rand_gen(0.0, 1.0)` is at most `accept_prob`.  The argument order is
`(lower, upper)`, as everywhere else in `qlat_ref.rng` and in qlat (it was
`(upper, lower)` before that rename; the values passed were swapped with it).
Passing `(1.0, 0.0)` would draw `1 - u` and flip every borderline decision --
at `accept_prob = 0.5` it disagrees with qlat on every seed -- so the order is
pinned by committed qlat decision vectors and by end-to-end chain flags, not by
a self-consistent recomputation.  `traj` is used only for the progress line,
not for the decision.

## The RNG Tree

The RNG splitting is qlat's and is part of the contract, because it decides
which configuration a given seed produces:

```
rs                                   driver seed
└── rs.split(str(traj))              one stream per trajectory
    ├── .split("set_rand_gauge_momentum")    the momentum draw
    └── .split("metropolis_accept")          the accept draw
```

The two consumers are split rather than shared **on purpose**: a rejected
trajectory must not disturb the momentum stream, so trajectory `n+1` draws the
same momentum whether or not trajectory `n` was accepted.

## Mutation Semantics

| Function | Mutates | Returns |
|---|---|---|
| `set_rand_gauge_momentum` | `gm` in place | `None` |
| `set_gm_force` | `gm_force` in place | `None` |
| `set_gm_force_dual` | `gm_force_dual` in place | `None` |
| `gf_evolve` | `gf` in place | `None` |
| `gf_evolve_dual` | `gf` in place | `None` |
| `gm_evolve_fg_pure_gauge` | `gm` only (never `gf_init`) | `None` |
| `gm_apply_gauge_transformation` | nothing | the new momentum |
| `gm_hamilton_node`, `gf_hamilton_node` | nothing | `float` |
| `metropolis_accept` | nothing | `(flag, accept_prob)` |
| `run_hmc_evolve_pure_gauge` | **both** `gm` and `gf` | `delta_h` |
| `run_hmc_pure_gauge` | `gf` in place **on accept** | `(flag, delta_h)` |

Because `run_hmc_pure_gauge` mutates its field, a caller that needs the
pre-trajectory configuration must take `gf.copy()` or `np.array(gf)` **before**
the call.  `np.asarray(gf)` is a live view, not a copy, and follows the update.

## Deviation: Single Process

qlat distributes the lattice over MPI nodes, so its `*_node` quantities are
node-local and its driver sums them with `glb_sum_double` and broadcasts the
accept/reject decision.  Here node-local **is** global:

- `gm_hamilton_node` and `gf_hamilton_node` return the global energies;
- `delta_h` is already the global energy violation;
- `metropolis_accept` draws on every process (there is only one) and returns
  the flag directly, with no broadcast.

The names keep qlat's `_node` suffix because they are qlat's names —
`gm_hamilton`/`gf_hamilton` do not exist in qlat's Python API.

There is also no communication: `set_gm_force` evaluates the staples by
shifting the single-process link array, which is exactly what qlat's
`refresh_expanded` boundary exchange makes available to its `_no_comm`
kernels.

The reference is **silent** where qlat prints a progress line per trajectory
(`displayln_info`): the test suite runs hundreds of trajectories.  The private
`qlat_ref.hmc_utils._VERBOSE` switch prints the same lines and changes nothing
else — no value and no RNG draw depends on it.

## Examples

```python
import qlat_ref as qr

geo = qr.Geometry(qr.Coordinate((2, 2, 2, 2)))
gf = qr.GaugeField(geo)
gf.set_unit()
ga = qr.GaugeAction(5.5, 0.0)

# one trajectory (mutates gf in place on accept)
flag, delta_h = qr.run_hmc_pure_gauge(gf, ga, 0, qr.RngState("traj-seed"))
print(flag, delta_h)

# the same trajectory, step by step, on an explicit copy
gf0 = gf.copy()
gm = qr.GaugeMomentum(geo)
qr.set_rand_gauge_momentum(gm, 1.0, qr.RngState("momentum-seed"))
delta_h = qr.run_hmc_evolve_pure_gauge(gm, gf, ga, 6, md_time=1.0)
flag, accept_prob = qr.metropolis_accept(delta_h, 0, qr.RngState("accept-seed"))
if flag:
    gf0 @= gf            # accept: keep the evolved field
else:
    gf @= gf0            # reject: restore
```

The `GaugeField`/`GaugeTransform`/`GaugeMomentum` algebra (`@=`, `*=`, `*`)
is qlat's, so `gf @= gf0` copies the field and `gf *= 0.5` scales it.

## Self-Checks

```bash
cd qlat-ref
python3 -m unittest discover -s tests -t . -v
```

The HMC tests are split by concern:

| File | Covers |
|---|---|
| `tests/test_hmc_momentum.py` | `set_rand_gauge_momentum` against the C++ vectors, the Gaussian distribution and the split layout |
| `tests/test_hmc_action.py` | `gf_hamilton_node` (plaquette and rectangle), the unit-field zero, `gm_hamilton_node` |
| `tests/test_hmc_force.py` | `set_gm_force`/`set_gm_force_dual` by finite differences, including the wrong-form negative controls |
| `tests/test_hmc_integrator.py` | `gf_evolve`/`gf_evolve_dual`, the `c1` rectangle term, unitarity after evolution |
| `tests/test_hmc_driver.py` | `metropolis_accept`, `gm_evolve_fg_pure_gauge`, `run_hmc_evolve_pure_gauge`, `run_hmc_pure_gauge`, reversibility and RNG-tree reproducibility |
| `tests/test_hmc_slow.py` | long multi-trajectory statistics; **opt-in** with `QLAT_REF_SLOW=1` |

The default suite runs in about half a minute and needs only `numpy`.  The slow
module is skipped unless `QLAT_REF_SLOW=1` is set:

```bash
cd qlat-ref
QLAT_REF_SLOW=1 python3 -m unittest tests.test_hmc_slow -v
```

Reference data for the deterministic parts is in `tests/data/`
(`hmc-vectors.json`, generated by `tests/data/make-hmc-vectors.py`), and
`tests/test_interface_parity.py` asserts that every HMC name exists in
`qlat.hmc`/`qlat.hmc_utils` with the same signature.
