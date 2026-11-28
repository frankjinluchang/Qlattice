# Canonical Gauge-Fixed HMC (`qlat_ref.cgfhmc`)

Canonical pure-gauge HMC with a **Landau gauge-fixing projection after every
molecular-dynamics step** and a **Fourier-mode-dependent partial momentum
refresh**.  It mirrors `qlat.cgfhmc`, so the names, argument order, defaults and
mutation semantics are `qlat`'s.

The plain trajectory driver it builds on is documented in
[`qlat_hmc.md`](qlat_hmc.md); this page covers only what CGFHMC adds.

## Algorithm

One trajectory is `run_hmc_pure_gauge` with two insertions per MD step, both at
the **same symmetric phase point** — strictly after the step's last integrator
sub-update and before the next step's first one:

1. **Landau gauge fixing** of both the links and the momentum.  The fixing does
   not mutate the field: `gt, info = fix_landau_gauge(gf0, ...)` returns a
   transformation with `gt * gf0` the fixed field, and the canonical update is

   $$
   U \leftarrow g\,U, \qquad P \leftarrow g\,P,
   \tag{1}
   $$

   where the second is the same-site adjoint action (no shift), so the momentum
   stays in the same gauge slice as the links.

2. **Partial momentum refresh**, per Fourier mode,

   $$
   P_k \leftarrow \sqrt{1-c(k)^2}\,P_k + c(k)\,\xi_k ,
   \tag{2}
   $$

   with $\xi$ a Gaussian momentum of the same ensemble.  See
   [below](#the-refresh).

The refresh deliberately replaces momentum, so its kinetic change must **not** be
charged to the Metropolis test.  With

$$
dK_{\text{sum}} = \sum_{\text{steps}} \bigl[ K_{\text{after}} - K_{\text{before}} \bigr],
\qquad K = \texttt{gm\_hamilton\_node}(P),
\tag{3}
$$

the accept variable is

$$
\Delta H_{\text{accept}} = \Delta H - dK_{\text{sum}} .
\tag{4}
$$

The **returned** `delta_h` is nevertheless the raw trajectory value
$H_{\text{final}} - H_{\text{initial}}$, exactly as `run_hmc_pure_gauge` defines
it, so the two drivers stay comparable.  `delta_h_accept` is
algorithm-internal and is reported only through `info_out`.

## The refresh

`c` is the **fresh-noise weight**, not the correlation:

| `c(k)` | meaning |
| --- | --- |
| `0` | the mode is frozen — no refresh at all |
| `1` | the mode is completely resampled |
| in between | retained autocorrelation $\sqrt{1-c^2}$ |

The design is a consequence of `c` being the *noise* fraction: both terms carry
the same per-mode variance, so each mode keeps it,

$$
E\lvert P'_k\rvert^2 = (1-c(k)^2)\sigma_m^2 + c(k)^2\sigma_m^2 = \sigma_m^2 ,
\tag{5}
$$

and the map is stationary for the Gaussian momentum ensemble.  Differing from
the uniform refresh only in the **mode-dependent decorrelation rate**, it lets
the ultraviolet modes — which dominate the force and the integrator error — be
decorrelated differently from the infrared ones.

A constant $c(k) = c$ reduces exactly to the uniform
(Ornstein–Uhlenbeck) law applied in real space, and that case is detected and
handled there directly: the mixing commutes with the (linear, orthonormal)
Fourier transform, so the result is mathematically identical and no transform is
performed.  Every other field is mixed mode by mode in Fourier space.
`info["path"]` records which route was taken.

The transform uses `numpy.fft` with `norm="ortho"` (unitary, a factor
$1/\sqrt{N}$ each way).  That is the only convention under which multiplying per
mode by $\sqrt{1-c^2}$ and $c$ preserves the variance; with any other
normalisation the noise and the momentum would carry different factors.

### The strict no-op

With $c(k) = 0$ everywhere the refresh **draws no random number** and leaves the
momentum bit-identical.  A frozen mode therefore costs nothing and cannot
disturb any stream.  This is a contract, not an optimisation:
`partial_refresh_momentum_mode` returns `(False, 0.0, info)` with
`info["no_op"] = True`, and the test suite compares the `RngState` before and
after to pin it.

## The c(k) field

`c` is a plain real `numpy` array of `float64`, shaped `(Lx, Ly, Lz, Lt)`, one
entry per Fourier mode.  There is no field type for it: it lives in Fourier
space, which no `qlat` field does.

!!! note "Two different axis orders in this package"
    The site axes of a field array are `(Nt, Nx, Ny, Nz)` — time first, as
    [`qlat_gauge.md`](qlat_gauge.md) documents for `U[mu, t, x, y, z, ...]`.  The
    `c(k)` field instead follows `Geometry::total_site`, which is
    `(Lx, Ly, Lz, Lt)`.  Do not "fix" one into the other: the mode grid must
    match `qlat`'s FFT mode indexing, and the site axes must match the field
    buffer.

### Mode grid and folding

Mode *indices* run over $n_i \in [0, L_i)$, but the *physical momentum*
attached to index $n$ is the folded (signed) one,

$$
k_i(n) = \frac{2\pi}{L_i} \times
\begin{array}{ll}
n_i & n_i \le L_i/2 \cr
n_i - L_i & \text{otherwise}
\end{array}
\tag{6}
$$

covering $[-\pi, \pi)$ in the FFT convention, with the Nyquist index $L_i/2$
included.

Folding is what makes a profile in $k^2$ physically meaningful: in a discrete
Fourier transform index $n$ and index $n - L$ denote the same momentum, so
without it a two-zone prescription would be applied to the wrong half of each
axis.  `k_vectors(geo)` returns the four components on this grid and
`k_squared(geo)` their sum of squares.

The *central-inversion partner* of index $n$ is $(-n) \bmod L$.  For one axis
that is `c[L-1-n]` reversed and then rolled by one — index `0` stays put and the
rest reverse — so the order is **flip then roll**.

### Why c must be real, even and bounded

$P_\mu(x) = i \sum_a \phi_a(x) T_a$ with the eight $\phi_a$ **real**, and the
component fields are transformed independently.  A real spectrum obeys
$X_{-k} = \overline{X_k}$, and mixing preserves that Hermitian symmetry only for
**even** coefficients,

$$
X'_{-k} = c(-k)X_{-k} + s(-k)\xi_{-k} = c(k)\overline{X_k} + s(k)\overline{\xi_k}
= \overline{X'_k}
\quad\text{iff } c(-k) = c(k),
\tag{7}
$$

with $s(k) = \sqrt{1-c(k)^2}$, which is even whenever $c$ is.  Odd coefficients
would make the inverse transform complex and break both anti-Hermiticity and
tracelessness.  For the same reason the scale factor must not be complex, so
$0 \le c \le 1$.

`regularize_c_field` enforces both conditions — **clipping** (not rescaling)
into $[0,1]$ so an out-of-range prescription is visibly saturated, and
**averaging** with the central-inversion partner, the least-squares projector
onto even fields.  NaN and infinity are refused by default rather than passed
through; `allow_nan="zero"` maps them to `0.0`, i.e. to a frozen mode, and
counts them.

Every consumer re-checks the invariants and raises rather than silently
repairing its input, so a hand-built field cannot be misapplied.

## API

```python
k_vectors(geo)                     # (4, Lx, Ly, Lz, Lt) float64 mode grid
k_squared(geo)                     # (Lx, Ly, Lz, Lt) float64, == sum(k_vectors**2, axis=0)
resolve_k_cut(geo, units)          # momentum threshold in units of 2*pi/Lx
regularize_c_field(c, *, in_place=False, allow_nan="raise")
generate_c_field(geo, func)        # call func(k_vectors(geo)) once and regularize
c_field_from_file(path)            # load a .npy field and regularize it
partial_refresh_momentum_mode(gm, rng, c_field)     # -> (applied, dK, info)
run_hmc_cgfhmc_pure_gauge(gf, ga, traj, rs, *, c_field,
                          is_reverse_test=False, n_step=6, md_time=1.0,
                          is_always_accept=False, gauge_fix_max_iter=25,
                          gauge_fix_eps=1.0e-10, gauge_fix_alpha=0.1,
                          info_out=None)            # -> (flag, delta_h)
```

The driver's first four parameters and the next four keyword-only ones are
`run_hmc_pure_gauge`'s, in the same order and with the same defaults, so the two
are interchangeable; the CGFHMC-specific parameters come after them.

`c_field` has **no default**: there is no prescription it could sensibly
assume.

`gauge_fix_max_iter=25` is deliberately much smaller than
`fix_landau_gauge`'s own default.  The field is re-fixed every MD step and
starts already near Landau gauge, so only the first step of a cold chain is
expensive.

`info_out`, when given, is filled in place with `dK_sum`,
`n_refresh_applied`, `n_step`, `n_fix`, `fix_n_iter_total`, `fix_n_iter_max`,
`fix_n_converged`, `fix_divergence_max`, `delta_h_raw` and `delta_h_accept`.
The return arity stays two.

### Example

```python
import numpy as np
import qlat_ref as qr
from qlat_ref import cgfhmc

geo = qr.Geometry(qr.Coordinate([16, 16, 16, 32]))
# Stiff infrared modes, aggressively refreshed ultraviolet ones.
threshold = cgfhmc.resolve_k_cut(geo, 1)
c = cgfhmc.generate_c_field(
    geo, lambda kv: np.where(np.sum(kv**2, axis=0) <= threshold**2, 0.3, 0.95)
)

gf = qr.GaugeField(geo)
gf.set_rand(qr.RngState("start"), 0.1, 2)
rs = qr.RngState("chain")
for traj in range(1, 11):
    info_out = {}
    flag, delta_h = cgfhmc.run_hmc_cgfhmc_pure_gauge(
        gf, qr.GaugeAction(6.0, 0.0), traj,
        rs.split("run_hmc_cgfhmc_pure_gauge"),
        c_field=c, n_step=64, md_time=4.0, info_out=info_out,
    )
    print(traj, flag, gf.plaq(), info_out["delta_h_accept"])
```

## RNG tree

The tree is `run_hmc_pure_gauge`'s with one extra consumer:

```
rs.split(str(traj))
├── .split("set_rand_gauge_momentum")             the momentum
├── .split("partial_refresh_momentum_base")       the refresh
│     └── .split(f"partial_refresh_momentum.{i}") per MD step i
└── .split("metropolis_accept")                   the accept draw
```

The streams are independent on purpose: a rejected trajectory does not disturb
the momentum or the refresh stream.

## Limitations

### Single process

`numpy.fft` is a global transform, so the refresh requires the whole lattice on
one node.  `qlat`'s `mk_fft` is a distributed FFT and runs under MPI, and its
example runs at `-n 2`.  The signatures and defaults are identical regardless;
only the parallel decomposition differs.

### `is_reverse_test` is not supported

The gauge-fixing projection is not part of the Hamiltonian flow, and it discards
the phase-space information a reverse test relies on: two gauge copies of the
same configuration fix to fields that are gauge-equivalent but *different*
representatives, because `gt` is not unique.  A CGFHMC trajectory is therefore
**not time reversible**.  Measured in this package, one trajectory forward with
`md_time = +1` and backward with `md_time = -1` returns to

| fixing | reverse $\lvert \Delta U \rvert$ | reverse $\lvert \Delta P \rvert$ |
| --- | --- | --- |
| off | `1.2e-15` | `2.9e-15` |
| on | `4.2e-01` | `7.8e-01` |

The keyword is kept so the signature matches `run_hmc_pure_gauge`, but passing
`True` raises `NotImplementedError` rather than returning a meaningless verdict.

## Tests

`tests/test_cgfhmc.py` pins every function — the mode grid and its folding, the
regularizer, the refresh laws and the driver — against independently computed
values and against the golden chains in `tests/data/cgfhmc-vectors.json`, which
was generated from a live `qlat` build.  Tolerances are absolute where a
quantity can legitimately be zero.

`tests/test_cgfhmc_slow.py` covers what only statistics or a larger lattice can
show: ensemble stationarity, chain health, that the mode dependence really
changes the trajectory, a lattice whose four extents all differ, and the
field/refresh invariants repeated on `4^4`.  It is opt-in with
`QLAT_REF_SLOW=1`.

```bash
cd qlat-ref
python3 -m pytest tests/test_cgfhmc.py
QLAT_REF_SLOW=1 python3 -m pytest tests/test_cgfhmc_slow.py
```

When a live `qlat` build is importable, `tests/test_qlat_parity.py` additionally
compares the two implementations value for value — the `c(k)` grid and its
folding, the regularizer, both refresh paths — and asserts that the
`is_reverse_test` message is byte-identical.  Those comparisons skip when
`qlat` is absent, and also when the installed `qlat` predates the CGFHMC module,
so this package stays testable with `numpy` alone.

The `c(k)` grid, the regularizer and the refresh are also pinned against the
committed golden chains, so an implementation error is caught without a live
`qlat`.
