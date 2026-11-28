#include <qlat/gauge-fixing.h>

#include <algorithm>
#include <cmath>

namespace qlat
{

namespace
{

const Int exp_order = 20;

// Component 0 of the scratch field in gf_functional_and_divergence() holds the
// per-site gauge-functional density, component 1 the per-site divergence
// density, so that one pass and one reduction produce both numbers.
const Int functional_and_divergence_multiplicity = 2;

Geometry get_expanded_geometry(const Geometry& geo)
{
  return geo_resize(geo, Coordinate(1, 1, 1, 1), Coordinate(1, 1, 1, 1));
}

array<Int, 4> get_dirs_arr(const std::vector<Int>& dirs)
{
  array<Int, 4> dirs_arr;
  dirs_arr.fill(0);
  for (Int i = 0; i < (Int)dirs.size(); ++i) {
    dirs_arr[i] = dirs[i];
  }
  return dirs_arr;
}

// Per-site density of the gauge functional: sum_{mu in dirs} Re Tr U_mu(x).
static qacc RealD gf_get_functional_density(const GaugeField& gf,
                                            const array<Int, 4>& dirs_arr,
                                            const Int n_dirs,
                                            const Coordinate& xl)
{
  const Vector<ColorMatrix> v = gf.get_elems_const(xl);
  RealD sum = 0.0;
  for (Int i = 0; i < n_dirs; ++i) {
    sum += matrix_trace(v[dirs_arr[i]]).real();
  }
  return sum;
}

// Per-site density of the divergence: Tr[D(x) D(x)^dagger].
static qacc RealD gf_get_divergence_density(const Field<ColorMatrix>& D,
                                            const Coordinate& xl)
{
  const ColorMatrix& d = D.get_elem(xl);
  RealD norm2 = 0.0;
  for (Int i = 0; i < 3; ++i) {
    for (Int j = 0; j < 3; ++j) {
      norm2 += qnorm(d(i, j));
    }
  }
  return norm2;
}

void gf_set_random_local_gradient(Field<ColorMatrix>& D, const GaugeField& gf,
                                  const GaugeField& gfe,
                                  const std::vector<Int>& dirs)
// gfe is the expanded copy of gf (neighbour links are available)
{
  const Int n_dirs = (Int)dirs.size();
  array<Int, 4> dirs_arr;
  dirs_arr.fill(0);
  for (Int i = 0; i < n_dirs; ++i) {
    dirs_arr[i] = dirs[i];
  }
  const Geometry& geo = gf.geo();
  D.init(geo, 1);
  qacc_for(index, geo.local_volume(), {
    const Geometry& geo = gf.geo();
    const Coordinate xl = geo.coordinate_from_index(index);
    const Vector<ColorMatrix> v = gf.get_elems_const(xl);
    ColorMatrix K;
    set_zero(K);
    for (Int i = 0; i < n_dirs; ++i) {
      const Int mu = dirs_arr[i];
      Coordinate xlm = xl;
      xlm[mu] -= 1;
      const Vector<ColorMatrix> vm = gfe.get_elems_const(xlm);
      K += v[mu] + matrix_adjoint(vm[mu]);
    }
    // D(x) = ((K^dagger - K) / 2) with the trace part removed
    D.get_elem(xl) = make_tr_less_anti_herm_matrix(matrix_adjoint(K));
  });
}

void gf_functional_and_divergence(RealD& functional, RealD& divergence,
                                  const GaugeField& gf,
                                  const Field<ColorMatrix>& D,
                                  const std::vector<Int>& dirs)
// Gauge functional of `gf` and divergence of the gradient `D`, in ONE pass over
// the sites and with ONE reduction.  Both are reduced quantities of the same
// geometry, so sharing the kernel and the allreduce removes one launch and one
// host reduction per sweep without touching the arithmetic: the per-site sums,
// their summation order and the reduction order are exactly those of
// gf_gauge_functional() and gf_gauge_divergence_from_gradient(), so the values
// are bit-identical.
{
  const Int n_dirs = (Int)dirs.size();
  const array<Int, 4> dirs_arr = get_dirs_arr(dirs);
  const Geometry& geo = gf.geo();
  Qassert(is_matching_geo(geo, D.geo()));
  Field<RealD> f;
  f.init(geo, functional_and_divergence_multiplicity);
  qacc_for(index, geo.local_volume(), {
    const Geometry& geo = gf.geo();
    const Coordinate xl = geo.coordinate_from_index(index);
    Vector<RealD> fv = f.get_elems(xl);
    // Component 0: gauge-functional density.
    fv[0] = gf_get_functional_density(gf, dirs_arr, n_dirs, xl);
    // Component 1: divergence density.
    fv[1] = gf_get_divergence_density(D, xl);
  });
  const std::vector<RealD> sums = field_glb_sum(f);
  functional = sums[0] / (geo.total_volume() * (RealD)n_dirs * 3.0);
  divergence = std::sqrt(sums[1] / (geo.total_volume() * 3.0));
}

void gf_update_gauge_field(GaugeField& gf, const Field<ColorMatrix>& g,
                           const Field<ColorMatrix>& ge)
// ge is the expanded copy of g (g(x+mu) is available)
//
// One pass over the sites, with the `mu` loop inside it: the update of the link
// U_mu(x) reads only g(x), U_mu(x) and ge(x+mu) = g(x+mu), so the four link
// directions are completely independent -- a site's `mu` link is never read or
// written by the update of another `mu`.  Fusing the four former per-`mu`
// passes into one therefore changes neither the result (bit-identical: each
// link still evaluates (g(x) U_mu(x)) g(x+mu)^dagger in the same order) nor
// the halo semantics, and it removes three kernel launches and three device
// synchronisations per sweep.
{
  const Geometry& geo = gf.geo();
  qacc_for(index, geo.local_volume(), {
    const Geometry& geo = gf.geo();
    const Coordinate xl = geo.coordinate_from_index(index);
    const ColorMatrix& gx = g.get_elem(xl);
    Vector<ColorMatrix> u = gf.get_elems(xl);
    for (Int mu = 0; mu < 4; ++mu) {
      Coordinate xlp = xl;
      xlp[mu] += 1;
      u[mu] = gx * u[mu] * matrix_adjoint(ge.get_elem(xlp));
    }
  });
}

}  // namespace

void gf_gauge_gradient(Field<ColorMatrix>& D, const GaugeField& gf,
                       const std::vector<Int>& dirs)
{
  TIMER("gf_gauge_gradient");
  GaugeField gfe;
  gfe.init(get_expanded_geometry(gf.geo()));
  gfe = gf;
  refresh_expanded(gfe);
  gf_set_random_local_gradient(D, gf, gfe, dirs);
}

RealD gf_gauge_functional(const GaugeField& gf, const std::vector<Int>& dirs)
{
  TIMER("gf_gauge_functional");
  const Int n_dirs = (Int)dirs.size();
  const array<Int, 4> dirs_arr = get_dirs_arr(dirs);
  const Geometry& geo = gf.geo();
  Field<RealD> f;
  f.init(geo, 1);
  qacc_for(index, geo.local_volume(), {
    const Geometry& geo = gf.geo();
    const Coordinate xl = geo.coordinate_from_index(index);
    f.get_elem(xl) = gf_get_functional_density(gf, dirs_arr, n_dirs, xl);
  });
  RealD sum = field_glb_sum(f)[0];
  return sum / (geo.total_volume() * (RealD)n_dirs * 3.0);
}

RealD gf_gauge_divergence_from_gradient(const Field<ColorMatrix>& D)
{
  TIMER("gf_gauge_divergence_from_gradient");
  const Geometry& geo = D.geo();
  Field<RealD> f;
  f.init(geo, 1);
  qacc_for(index, geo.local_volume(), {
    const Geometry& geo = D.geo();
    const Coordinate xl = geo.coordinate_from_index(index);
    f.get_elem(xl) = gf_get_divergence_density(D, xl);
  });
  const RealD sum = field_glb_sum(f)[0];
  return std::sqrt(sum / (geo.total_volume() * 3.0));
}

RealD gf_gauge_divergence(const GaugeField& gf, const std::vector<Int>& dirs)
{
  Field<ColorMatrix> D;
  gf_gauge_gradient(D, gf, dirs);
  return gf_gauge_divergence_from_gradient(D);
}

RealD gf_gauge_unitarity_error(const Field<ColorMatrix>& f)
{
  TIMER("gf_gauge_unitarity_error");
  const Geometry& geo = f.geo();
  Field<RealD> f_err;
  f_err.init(geo, 1);
  qacc_for(index, geo.local_volume(), {
    const Geometry& geo = f.geo();
    const Coordinate xl = geo.coordinate_from_index(index);
    const Vector<ColorMatrix> v = f.get_elems_const(xl);
    const Int multiplicity = f.multiplicity;
    RealD err = 0.0;
    for (Int m = 0; m < multiplicity; ++m) {
      const ColorMatrix mm = v[m] * matrix_adjoint(v[m]);
      for (Int i = 0; i < 3; ++i) {
        for (Int j = 0; j < 3; ++j) {
          const ComplexD delta =
              mm(i, j) - (i == j ? ComplexD(1.0) : ComplexD(0.0));
          // std::abs(ComplexD) is a host-only std::complex overload that is not
          // device callable inside this qacc_for CUDA lambda;
          // std::sqrt(qnorm(delta)) is the device-safe equivalent (qnorm is the
          // squared magnitude under QLAT_USE_ACC).
          err = std::max(err, std::sqrt(qnorm(delta)));
        }
      }
    }
    f_err.get_elem(xl) = err;
  });
  return field_glb_max(f_err)[0];
}

GaugeFixingInfo gf_fix_gauge(GaugeTransform& gt, const GaugeField& gf,
                             const std::vector<Int>& dirs, const Int max_iter,
                             const RealD eps, const RealD alpha,
                             const bool verbose)
{
  TIMER("gf_fix_gauge");
  Qassert(gf.initialized);
  Qassert(dirs.size() > 0);
  for (Int i = 0; i < (Int)dirs.size(); ++i) {
    Qassert(0 <= dirs[i] and dirs[i] < 4);
  }
  const Geometry& geo = gf.geo();
  const Geometry geo_expanded = get_expanded_geometry(geo);
  // The expanded field is the running configuration; the const input `gf` is
  // only read once, when seeding the compact region below.
  GaugeField gfe;
  gfe.init(geo_expanded);
  gfe = gf;
  refresh_expanded(gfe);
  // Accumulated gauge transformation: (gt * gf) is the gauge-fixed field.
  gt.init(geo);
  set_unit(gt);
  Field<ColorMatrix> D;
  // `D` must be on the expanded geometry: gf_set_random_local_gradient() calls
  // D.init(gf.geo(), 1) and its `gf` argument is now the expanded `gfe`, so a
  // compact `D` would trip Field::init's compact -> expanded Qassert.
  D.init(geo_expanded, 1);
  GaugeTransform gt_step;
  gt_step.init(geo);
  Field<ColorMatrix> ge;
  ge.init(geo_expanded, 1);
  GaugeFixingInfo info;
  info.converged = false;
  info.n_iter = 0;
  info.max_iter = max_iter;
  info.eps = eps;
  info.alpha = alpha;
  info.functional = 0.0;
  info.divergence = 0.0;
  for (Int n_iter = 1; n_iter <= max_iter; ++n_iter) {
    // The halo of `gfe` is valid here: refreshed before the loop for sweep 1,
    // at the bottom of the loop body for the following sweeps.
    gf_set_random_local_gradient(D, gfe, gfe, dirs);
    // The functional is part of the reported history (info.functional_list),
    // and the convergence test below needs the divergence; both are per-site
    // densities reduced over the same geometry, so they share one pass and one
    // reduction.  This is the only evaluation of the functional in the loop.
    RealD functional;
    RealD divergence;
    gf_functional_and_divergence(functional, divergence, gfe, D, dirs);
    info.functional_list.push_back(functional);
    info.divergence_list.push_back(divergence);
    info.n_iter = n_iter;
    if (verbose) {
      displayln_info(
          ssprintf("gf_fix_gauge: sweep %d: functional = %.15e;"
                   " divergence = %.3e",
                   n_iter, functional, divergence));
    }
    if (divergence <= eps) {
      info.converged = true;
      break;
    }
    qacc_for(index, geo.local_volume(), {
      const Geometry& geo = gfe.geo();
      const Coordinate xl = geo.coordinate_from_index(index);
      const ColorMatrix h = (alpha * D.get_elem(xl));
      gt_step.get_elem(xl) = make_matrix_exp(h, exp_order);
    });
    // gt <- gt_step * gt (so that gt is the transform of all sweeps so far).
    gt_apply_gauge_transformation(gt, gt_step);
    ge = gt_step;
    refresh_expanded(ge);
    gf_update_gauge_field(gfe, gt_step, ge);
    // The halo of `gfe` went stale in the update above (compact region only).
    refresh_expanded(gfe);
  }
  // Final info is computed on the working field, reusing `D`: no extra
  // expanded copy is allocated (gf_gauge_gradient() would allocate its own).
  gf_set_random_local_gradient(D, gfe, gfe, dirs);
  gf_functional_and_divergence(info.functional, info.divergence, gfe, D, dirs);
  return info;
}

GaugeFixingInfo gf_fix_landau_gauge(GaugeTransform& gt, const GaugeField& gf,
                                    const Int max_iter, const RealD eps,
                                    const RealD alpha, const bool verbose)
{
  const std::vector<Int> dirs = {0, 1, 2, 3};
  return gf_fix_gauge(gt, gf, dirs, max_iter, eps, alpha, verbose);
}

GaugeFixingInfo gf_fix_coulomb_gauge(GaugeTransform& gt, const GaugeField& gf,
                                     const Int max_iter, const RealD eps,
                                     const RealD alpha, const bool verbose)
{
  const std::vector<Int> dirs = {1, 2, 3};
  return gf_fix_gauge(gt, gf, dirs, max_iter, eps, alpha, verbose);
}

}  // namespace qlat
