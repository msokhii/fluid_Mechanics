// ------------------------------------------------------------
// 2D incompressible Navier–Stokes (lid-driven cavity) on a MAC grid
// Projection (fractional-step) method:
//   1) predict u*, v* (advection + diffusion, no pressure)
//   2) solve Poisson: Lap(phi) = (1/dt) div(u*,v*)
//   3) correct: u^{n+1} = u* - dt * grad(phi),  p += phi
//
// Poisson solvers to compare (single-grid):
//   - Jacobi
//   - Gauss–Seidel/SOR
//
// Outputs:
//   - Summary timing + iterations
//   - Optional VTK for ParaView
//   - Optional centerline CSV (u at x=0.5, v at y=0.5)
//   - Optional Ghia sample-point CSVs (exact Ghia Table sample coordinates)
//
// Build:
//   g++ -O3 -march=native -std=c++17 -DNDEBUG ns_cavity_project.cpp -o cavity
//
// Fast-ish test run (recommended while debugging/timing):
//   ./cavity --solver sor --Nx 128 --Ny 128 --Re 400 --steps 800 --warmup 200 --noVtk \
//            --tol 1e-4 --maxIters 2000 --checkEvery 25 --omega 1.9
//
// Full run with outputs:
//   ./cavity --solver sor --Nx 128 --Ny 128 --Re 400 --steps 5000 --vtkEvery 200 \
//            --centerline --ghia --prefix N128_sor
//
// Notes:
// - For fair Jacobi vs SOR comparisons: same Nx,Ny, Re, dt policy, tol, maxIters, checkEvery.
// - Disable VTK while measuring performance (I/O dominates).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

#if defined(__x86_64__) || defined(_M_X64)
  #include <x86intrin.h>
  static inline uint64_t rdtsc_serialized() {
    unsigned aux = 0;
    return __rdtscp(&aux); // partially serializing
  }
#else
  static inline uint64_t rdtsc_serialized() { return 0; }
#endif

struct Field {
  int nx = 0, ny = 0;
  std::vector<double> a;

  Field() = default;
  Field(int nx_, int ny_, double val = 0.0) { resize(nx_, ny_, val); }

  void resize(int nx_, int ny_, double val = 0.0) {
    nx = nx_;
    ny = ny_;
    a.assign((size_t)nx * (size_t)ny, val);
  }

  inline double &operator()(int i, int j) {
    return a[(size_t)i + (size_t)nx * (size_t)j];
  }
  inline double operator()(int i, int j) const {
    return a[(size_t)i + (size_t)nx * (size_t)j];
  }

  void fill(double val) { std::fill(a.begin(), a.end(), val); }
};

struct Timer {
  using clock = std::chrono::steady_clock;
  clock::time_point t0;
  void start() { t0 = clock::now(); }
  double seconds() const {
    return std::chrono::duration<double>(clock::now() - t0).count();
  }
};

// ------------------------------ Poisson tools ------------------------------
static inline void apply_neumann_bc(Field &q, int Nx, int Ny) {
  // q is (Nx+2) x (Ny+2), interior i=1..Nx, j=1..Ny
  for (int j = 1; j <= Ny; ++j) {
    q(0, j)    = q(1, j);
    q(Nx+1, j) = q(Nx, j);
  }
  for (int i = 1; i <= Nx; ++i) {
    q(i, 0)    = q(i, 1);
    q(i, Ny+1) = q(i, Ny);
  }
  // corners
  q(0, 0)       = q(1, 1);
  q(0, Ny+1)    = q(1, Ny);
  q(Nx+1, 0)    = q(Nx, 1);
  q(Nx+1, Ny+1) = q(Nx, Ny);
}

static inline void remove_mean(Field &q, int Nx, int Ny) {
  // Neumann Poisson is defined up to an additive constant: fix by mean-zero.
  double sum = 0.0;
  for (int j = 1; j <= Ny; ++j)
    for (int i = 1; i <= Nx; ++i)
      sum += q(i, j);

  double mean = sum / (double)(Nx * Ny);

  for (int j = 1; j <= Ny; ++j)
    for (int i = 1; i <= Nx; ++i)
      q(i, j) -= mean;
}

static inline double poisson_residual_inf(const Field &x, const Field &b,
                                          int Nx, int Ny, double dx, double dy) {
  // r = b - Lap(x)
  const double invdx2 = 1.0 / (dx * dx);
  const double invdy2 = 1.0 / (dy * dy);
  double rinf = 0.0;

  for (int j = 1; j <= Ny; ++j) {
    for (int i = 1; i <= Nx; ++i) {
      double lap = (x(i+1, j) - 2.0*x(i, j) + x(i-1, j)) * invdx2
                 + (x(i, j+1) - 2.0*x(i, j) + x(i, j-1)) * invdy2;
      double r = b(i, j) - lap;
      rinf = std::max(rinf, std::abs(r));
    }
  }
  return rinf;
}

struct PoissonStats {
  int iters = 0;
  double res_inf = 0.0;     // last computed residual inf-norm
  double max_delta = 0.0;   // last sweep's max update magnitude
};

struct PoissonSolver {
  virtual ~PoissonSolver() = default;
  virtual std::string name() const = 0;

  // Solve Lap(phi)=rhs (Neumann BC), using:
  // - tol on residual inf-norm (checked only every checkEvery iterations)
  // - optional deltaTol early-stop if >0 (checked every iter)
  virtual PoissonStats solve(Field &phi, const Field &rhs,
                             int Nx, int Ny, double dx, double dy,
                             int maxIters, double tol,
                             int checkEvery,
                             double deltaTol) = 0;
};

// ------------------------------ Jacobi ------------------------------
struct PoissonJacobi final : public PoissonSolver {
  std::string name() const override { return "jacobi"; }

  PoissonStats solve(Field &phi, const Field &rhs,
                     int Nx, int Ny, double dx, double dy,
                     int maxIters, double tol,
                     int checkEvery,
                     double deltaTol) override {
    Field next(phi.nx, phi.ny, 0.0);

    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double denom  = 2.0 * (invdx2 + invdy2);

    apply_neumann_bc(phi, Nx, Ny);
    remove_mean(phi, Nx, Ny);

    double rinf = poisson_residual_inf(phi, rhs, Nx, Ny, dx, dy);
    double maxDelta = 0.0;
    int it = 0;

    for (int k = 1; k <= maxIters; ++k) {
      it = k;
      apply_neumann_bc(phi, Nx, Ny);

      // compute next
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          next(i, j) =
            ( (phi(i+1, j) + phi(i-1, j)) * invdx2
            + (phi(i, j+1) + phi(i, j-1)) * invdy2
            - rhs(i, j) ) / denom;
        }
      }

      // update + track maxDelta
      maxDelta = 0.0;
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          double d = std::abs(next(i, j) - phi(i, j));
          maxDelta = std::max(maxDelta, d);
          phi(i, j) = next(i, j);
        }
      }

      remove_mean(phi, Nx, Ny);
      apply_neumann_bc(phi, Nx, Ny);

      // cheap early stop if requested
      if (deltaTol > 0.0 && maxDelta < deltaTol) {
        break;
      }

      // expensive check only occasionally
      if (checkEvery > 0 && (it % checkEvery == 0)) {
        rinf = poisson_residual_inf(phi, rhs, Nx, Ny, dx, dy);
        if (rinf < tol) break;
      }
    }

    // ensure residual is available for reporting
    rinf = poisson_residual_inf(phi, rhs, Nx, Ny, dx, dy);
    apply_neumann_bc(phi, Nx, Ny);

    return PoissonStats{it, rinf, maxDelta};
  }
};

// ------------------------------ SOR ------------------------------
struct PoissonSOR final : public PoissonSolver {
  double omega = 1.7;
  explicit PoissonSOR(double w) : omega(w) {}
  std::string name() const override { return "sor"; }

  PoissonStats solve(Field &phi, const Field &rhs,
                     int Nx, int Ny, double dx, double dy,
                     int maxIters, double tol,
                     int checkEvery,
                     double deltaTol) override {
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double denom  = 2.0 * (invdx2 + invdy2);

    apply_neumann_bc(phi, Nx, Ny);
    remove_mean(phi, Nx, Ny);

    double rinf = poisson_residual_inf(phi, rhs, Nx, Ny, dx, dy);
    double maxDelta = 0.0;
    int it = 0;

    for (int k = 1; k <= maxIters; ++k) {
      it = k;
      apply_neumann_bc(phi, Nx, Ny);

      maxDelta = 0.0;
      for (int j = 1; j <= Ny; ++j) {
        for (int i = 1; i <= Nx; ++i) {
          double xold = phi(i, j);
          double xnew =
            ( (phi(i+1, j) + phi(i-1, j)) * invdx2
            + (phi(i, j+1) + phi(i, j-1)) * invdy2
            - rhs(i, j) ) / denom;

          double xr = (1.0 - omega) * xold + omega * xnew;
          maxDelta = std::max(maxDelta, std::abs(xr - xold));
          phi(i, j) = xr;
        }
      }

      remove_mean(phi, Nx, Ny);
      apply_neumann_bc(phi, Nx, Ny);

      if (deltaTol > 0.0 && maxDelta < deltaTol) {
        break;
      }

      if (checkEvery > 0 && (it % checkEvery == 0)) {
        rinf = poisson_residual_inf(phi, rhs, Nx, Ny, dx, dy);
        if (rinf < tol) break;
      }
    }

    rinf = poisson_residual_inf(phi, rhs, Nx, Ny, dx, dy);
    apply_neumann_bc(phi, Nx, Ny);

    return PoissonStats{it, rinf, maxDelta};
  }
};

// ------------------------------ Simulation ------------------------------
struct SimConfig {
  int Nx = 128, Ny = 128;
  double Lx = 1.0, Ly = 1.0;
  double U_lid = 1.0;
  double Re = 400.0;

  // output directories
  std::string vtkDir = "vtk_out";
  std::string csvDir = "csv_out";

  int steps = 5000;
  int vtkEvery = 200;
  bool writeVtk = true;

  // timestep control
  double dtMax = 0.01;
  double cfl = 0.5;
  bool fixedDt = false;
  double dtFixed = 0.005;

  // Poisson controls
  int poissonMaxIters = 8000;
  double poissonTol = 1e-6;     // residual inf-norm tolerance
  int poissonCheckEvery = 25;   // compute residual every K iterations
  double poissonDeltaTol = 0.0; // optional early stop on update size; 0 disables

  double sorOmega = 1.7;

  // timing
  int warmupSteps = 200; // not included in avg iters/res/timings

  // outputs
  bool writeCenterline = false;
  bool writeGhia = false;
  std::string prefix = "run";
};

struct RunStats {
  // averaged over timed steps only (steps > warmupSteps)
  double avgStepSec = 0.0;
  double avgPoissonSec = 0.0;
  double avgStepCycles = 0.0;
  double avgPoissonCycles = 0.0;

  double avgPoissonIters = 0.0;
  double avgPoissonResInf = 0.0;
  double avgPoissonMaxDelta = 0.0;

  double maxDiv = 0.0; // after final step
};

struct Simulation {
  SimConfig cfg;
  double dx, dy, nu;

  // cell-centered fields (Nx+2)x(Ny+2)
  Field p, phi, rhs;

  // MAC fields:
  // u: (Nx+1)x(Ny+2), i=0..Nx, j=0..Ny+1
  // v: (Nx+2)x(Ny+1), i=0..Nx+1, j=0..Ny
  Field u, v, us, vs;

  PoissonSolver *ps = nullptr;

  explicit Simulation(const SimConfig &c, PoissonSolver *solver)
      : cfg(c), ps(solver) {
    dx = cfg.Lx / cfg.Nx;
    dy = cfg.Ly / cfg.Ny;
    nu = cfg.U_lid * cfg.Lx / cfg.Re;

    p.resize(cfg.Nx + 2, cfg.Ny + 2, 0.0);
    phi.resize(cfg.Nx + 2, cfg.Ny + 2, 0.0);
    rhs.resize(cfg.Nx + 2, cfg.Ny + 2, 0.0);

    u.resize(cfg.Nx + 1, cfg.Ny + 2, 0.0);
    v.resize(cfg.Nx + 2, cfg.Ny + 1, 0.0);
    us.resize(cfg.Nx + 1, cfg.Ny + 2, 0.0);
    vs.resize(cfg.Nx + 2, cfg.Ny + 1, 0.0);

    // Create output folders automatically
    if (cfg.writeVtk) {
      fs::create_directories(cfg.vtkDir);
    }
    if (cfg.writeCenterline || cfg.writeGhia) {
      fs::create_directories(cfg.csvDir);
    }
  }

  void apply_velocity_bc(Field &uu, Field &vv) {
    const int Nx = cfg.Nx;
    const int Ny = cfg.Ny;

    // u left/right walls (faces)
    for (int j = 1; j <= Ny; ++j) {
      uu(0, j)  = 0.0;
      uu(Nx, j) = 0.0;
    }
    // bottom wall u=0 via ghost reflect
    for (int i = 0; i <= Nx; ++i) {
      uu(i, 0) = -uu(i, 1);
    }
    // top wall u=U_lid via ghost reflect
    for (int i = 0; i <= Nx; ++i) {
      uu(i, Ny + 1) = 2.0 * cfg.U_lid - uu(i, Ny);
    }

    // v bottom/top walls (faces lie on walls)
    for (int i = 1; i <= Nx; ++i) {
      vv(i, 0)  = 0.0;
      vv(i, Ny) = 0.0;
    }
    // v left/right walls via ghost reflect
    for (int j = 0; j <= Ny; ++j) {
      vv(0, j)    = -vv(1, j);
      vv(Nx+1, j) = -vv(Nx, j);
    }
  }

  inline double v_at_u(int i, int j, const Field &vv) const {
    // u(i,j) at (x=i*dx, y=(j-0.5)dy)
    return 0.25 * (vv(i, j) + vv(i+1, j) + vv(i, j-1) + vv(i+1, j-1));
  }

  inline double u_at_v(int i, int j, const Field &uu) const {
    // v(i,j) at (x=(i-0.5)dx, y=j*dy)
    return 0.25 * (uu(i, j) + uu(i, j+1) + uu(i-1, j) + uu(i-1, j+1));
  }

  inline double ddx_upwind(double qm1, double q0, double qp1, double vel, double h) const {
    return (vel > 0.0) ? (q0 - qm1) / h : (qp1 - q0) / h;
  }

  inline double ddy_upwind(double qm1, double q0, double qp1, double vel, double h) const {
    return (vel > 0.0) ? (q0 - qm1) / h : (qp1 - q0) / h;
  }

  double compute_dt() const {
    if (cfg.fixedDt) return cfg.dtFixed;

    double umax = 0.0;
    for (int j = 1; j <= cfg.Ny; ++j)
      for (int i = 0; i <= cfg.Nx; ++i)
        umax = std::max(umax, std::abs(u(i, j)));
    for (int j = 0; j <= cfg.Ny; ++j)
      for (int i = 1; i <= cfg.Nx; ++i)
        umax = std::max(umax, std::abs(v(i, j)));

    double hmin = std::min(dx, dy);
    double dt_adv  = (umax > 1e-12) ? cfg.cfl * hmin / umax : cfg.dtMax;
    double dt_diff = (nu > 0.0) ? 0.25 * (hmin * hmin) / nu : cfg.dtMax;
    return std::min(cfg.dtMax, std::min(dt_adv, dt_diff));
  }

  double max_divergence() const {
    double md = 0.0;
    for (int j = 1; j <= cfg.Ny; ++j) {
      for (int i = 1; i <= cfg.Nx; ++i) {
        double div = (u(i, j) - u(i-1, j)) / dx
                   + (v(i, j) - v(i, j-1)) / dy;
        md = std::max(md, std::abs(div));
      }
    }
    return md;
  }

  void write_vtk(int frame) const {
    if (!cfg.writeVtk) return;

    // Ensure directory exists (safe if already exists)
    fs::create_directories(cfg.vtkDir);

    std::ostringstream fn;
    fn << cfg.vtkDir << "/out_" << std::setw(6) << std::setfill('0') << frame << ".vtk";
    std::ofstream out(fn.str());
    if (!out) return;

    out << "# vtk DataFile Version 2.0\n";
    out << "Lid-driven cavity\n";
    out << "ASCII\n";
    out << "DATASET STRUCTURED_POINTS\n";
    out << "DIMENSIONS " << cfg.Nx << " " << cfg.Ny << " 1\n";
    out << "ORIGIN " << 0.5 * dx << " " << 0.5 * dy << " 0\n";
    out << "SPACING " << dx << " " << dy << " 1\n";
    out << "POINT_DATA " << (cfg.Nx * cfg.Ny) << "\n";

    out << "SCALARS pressure double 1\n";
    out << "LOOKUP_TABLE default\n";
    for (int j = 1; j <= cfg.Ny; ++j)
      for (int i = 1; i <= cfg.Nx; ++i)
        out << p(i, j) << "\n";

    out << "VECTORS velocity double\n";
    for (int j = 1; j <= cfg.Ny; ++j) {
      for (int i = 1; i <= cfg.Nx; ++i) {
        double uc = 0.5 * (u(i, j) + u(i-1, j));
        double vc = 0.5 * (v(i, j) + v(i, j-1));
        out << uc << " " << vc << " 0\n";
      }
    }
  }

  // cell-centered u,v from MAC
  inline double uc(int i, int j) const { return 0.5 * (u(i, j) + u(i-1, j)); }
  inline double vc(int i, int j) const { return 0.5 * (v(i, j) + v(i, j-1)); }

  double sample_uc(double x, double y) const {
    // bilinear on cell centers
    const int Nx = cfg.Nx, Ny = cfg.Ny;
    double iReal = x / dx + 0.5;
    double jReal = y / dy + 0.5;

    int i0 = (int)std::floor(iReal);
    int j0 = (int)std::floor(jReal);
    i0 = std::max(1, std::min(Nx - 1, i0));
    j0 = std::max(1, std::min(Ny - 1, j0));
    int i1 = i0 + 1, j1 = j0 + 1;

    double x0 = (i0 - 0.5) * dx;
    double y0 = (j0 - 0.5) * dy;
    double wx = (dx > 0) ? (x - x0) / dx : 0.0;
    double wy = (dy > 0) ? (y - y0) / dy : 0.0;
    wx = std::max(0.0, std::min(1.0, wx));
    wy = std::max(0.0, std::min(1.0, wy));

    double u00 = uc(i0, j0), u10 = uc(i1, j0);
    double u01 = uc(i0, j1), u11 = uc(i1, j1);
    double ux0 = (1.0 - wx) * u00 + wx * u10;
    double ux1 = (1.0 - wx) * u01 + wx * u11;
    return (1.0 - wy) * ux0 + wy * ux1;
  }

  double sample_vc(double x, double y) const {
    const int Nx = cfg.Nx, Ny = cfg.Ny;
    double iReal = x / dx + 0.5;
    double jReal = y / dy + 0.5;

    int i0 = (int)std::floor(iReal);
    int j0 = (int)std::floor(jReal);
    i0 = std::max(1, std::min(Nx - 1, i0));
    j0 = std::max(1, std::min(Ny - 1, j0));
    int i1 = i0 + 1, j1 = j0 + 1;

    double x0 = (i0 - 0.5) * dx;
    double y0 = (j0 - 0.5) * dy;
    double wx = (dx > 0) ? (x - x0) / dx : 0.0;
    double wy = (dy > 0) ? (y - y0) / dy : 0.0;
    wx = std::max(0.0, std::min(1.0, wx));
    wy = std::max(0.0, std::min(1.0, wy));

    double v00 = vc(i0, j0), v10 = vc(i1, j0);
    double v01 = vc(i0, j1), v11 = vc(i1, j1);
    double vx0 = (1.0 - wx) * v00 + wx * v10;
    double vx1 = (1.0 - wx) * v01 + wx * v11;
    return (1.0 - wy) * vx0 + wy * vx1;
  }

  void write_centerlines_csv() const {
    fs::create_directories(cfg.csvDir);

    // u(x=0.5,y) on Ny points (cell centers), v(x,y=0.5) on Nx points
    {
      std::ofstream out(cfg.csvDir + "/" + cfg.prefix + "_u_x0p5.csv");
      if (out) {
        out << "y,u\n";
        for (int j = 1; j <= cfg.Ny; ++j) {
          double y = (j - 0.5) * dy;
          double uval = sample_uc(0.5 * cfg.Lx, y);
          out << std::setprecision(16) << y << "," << uval << "\n";
        }
      }
    }
    {
      std::ofstream out(cfg.csvDir + "/" + cfg.prefix + "_v_y0p5.csv");
      if (out) {
        out << "x,v\n";
        for (int i = 1; i <= cfg.Nx; ++i) {
          double x = (i - 0.5) * dx;
          double vval = sample_vc(x, 0.5 * cfg.Ly);
          out << std::setprecision(16) << x << "," << vval << "\n";
        }
      }
    }
  }

  void write_ghia_csv() const {
    fs::create_directories(cfg.csvDir);

    // Standard Ghia sample coordinates (Table I for u at x=0.5, Table II for v at y=0.5)
    static const double yPts[] = {
      1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
      0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000
    };
    static const double xPts[] = {
      1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
      0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000
    };

    {
      std::ofstream out(cfg.csvDir + "/" + cfg.prefix + "_ghia_u.csv");
      if (out) {
        out << "y,u_sim\n";
        for (double y : yPts) {
          double uval;
          if (y <= 0.0) uval = 0.0;
          else if (y >= 1.0) uval = cfg.U_lid;
          else uval = sample_uc(0.5 * cfg.Lx, y * cfg.Ly);
          out << std::setprecision(16) << y << "," << uval << "\n";
        }
      }
    }
    {
      std::ofstream out(cfg.csvDir + "/" + cfg.prefix + "_ghia_v.csv");
      if (out) {
        out << "x,v_sim\n";
        for (double x : xPts) {
          double vval;
          if (x <= 0.0 || x >= 1.0) vval = 0.0;
          else vval = sample_vc(x * cfg.Lx, 0.5 * cfg.Ly);
          out << std::setprecision(16) << x << "," << vval << "\n";
        }
      }
    }
  }

  // One step. Returns PoissonStats; also reports Poisson timing/cycles.
  PoissonStats step_once(double &dt,
                         double &poissonSec, uint64_t &poissonCycles) {
    apply_velocity_bc(u, v);
    dt = compute_dt();

    // ---- Predict u* ----
    for (int j = 1; j <= cfg.Ny; ++j) {
      for (int i = 1; i <= cfg.Nx - 1; ++i) {
        double u0 = u(i, j);
        double v0 = v_at_u(i, j, v);

        double dudx = ddx_upwind(u(i-1, j), u0, u(i+1, j), u0, dx);
        double dudy = ddy_upwind(u(i, j-1), u0, u(i, j+1), v0, dy);
        double adv  = u0 * dudx + v0 * dudy;

        double lap = (u(i+1, j) - 2.0*u0 + u(i-1, j)) / (dx*dx)
                   + (u(i, j+1) - 2.0*u0 + u(i, j-1)) / (dy*dy);

        us(i, j) = u0 + dt * (-adv + nu * lap);
      }
    }
    for (int j = 1; j <= cfg.Ny; ++j) {
      us(0, j) = u(0, j);
      us(cfg.Nx, j) = u(cfg.Nx, j);
    }

    // ---- Predict v* ----
    for (int j = 1; j <= cfg.Ny - 1; ++j) {
      for (int i = 1; i <= cfg.Nx; ++i) {
        double v0 = v(i, j);
        double u0 = u_at_v(i, j, u);

        double dvdx = ddx_upwind(v(i-1, j), v0, v(i+1, j), u0, dx);
        double dvdy = ddy_upwind(v(i, j-1), v0, v(i, j+1), v0, dy);
        double adv  = u0 * dvdx + v0 * dvdy;

        double lap = (v(i+1, j) - 2.0*v0 + v(i-1, j)) / (dx*dx)
                   + (v(i, j+1) - 2.0*v0 + v(i, j-1)) / (dy*dy);

        vs(i, j) = v0 + dt * (-adv + nu * lap);
      }
    }
    for (int i = 1; i <= cfg.Nx; ++i) {
      vs(i, 0) = v(i, 0);
      vs(i, cfg.Ny) = v(i, cfg.Ny);
    }
    for (int j = 0; j <= cfg.Ny; ++j) {
      vs(0, j) = v(0, j);
      vs(cfg.Nx+1, j) = v(cfg.Nx+1, j);
    }

    apply_velocity_bc(us, vs);

    // ---- RHS = (1/dt) div(u*,v*) ----
    rhs.fill(0.0);
    for (int j = 1; j <= cfg.Ny; ++j) {
      for (int i = 1; i <= cfg.Nx; ++i) {
        double div = (us(i, j) - us(i-1, j)) / dx
                   + (vs(i, j) - vs(i, j-1)) / dy;
        rhs(i, j) = div / dt;
      }
    }

    // ---- Solve Poisson ----
    phi.fill(0.0);
    Timer tP; tP.start();
    uint64_t c0 = rdtsc_serialized();

    PoissonStats pst = ps->solve(
      phi, rhs, cfg.Nx, cfg.Ny, dx, dy,
      cfg.poissonMaxIters, cfg.poissonTol,
      cfg.poissonCheckEvery, cfg.poissonDeltaTol
    );

    uint64_t c1 = rdtsc_serialized();
    poissonCycles += (c1 - c0);
    poissonSec += tP.seconds();

    // ---- Correct velocities ----
    for (int j = 1; j <= cfg.Ny; ++j) {
      for (int i = 1; i <= cfg.Nx - 1; ++i) {
        us(i, j) -= dt * (phi(i+1, j) - phi(i, j)) / dx;
      }
    }
    for (int j = 1; j <= cfg.Ny - 1; ++j) {
      for (int i = 1; i <= cfg.Nx; ++i) {
        vs(i, j) -= dt * (phi(i, j+1) - phi(i, j)) / dy;
      }
    }

    // pressure update
    for (int j = 1; j <= cfg.Ny; ++j)
      for (int i = 1; i <= cfg.Nx; ++i)
        p(i, j) += phi(i, j);

    apply_velocity_bc(us, vs);

    u.a.swap(us.a);
    v.a.swap(vs.a);

    return pst;
  }

  RunStats run() {
    RunStats st;
    apply_velocity_bc(u, v);

    int frame = 0;
    if (cfg.writeVtk) write_vtk(frame++);

    // timed accumulators (exclude warmup)
    double stepSecSum = 0.0, poissonSecSum = 0.0;
    uint64_t stepCycSum = 0, poissonCycSum = 0;
    double itSum = 0.0, resSum = 0.0, dltSum = 0.0;
    int timed = 0;

    for (int n = 1; n <= cfg.steps; ++n) {
      Timer tS; tS.start();
      uint64_t s0 = rdtsc_serialized();

      double dt = 0.0;
      double poissonSecLocal = 0.0;
      uint64_t poissonCycLocal = 0;

      PoissonStats pst = step_once(dt, poissonSecLocal, poissonCycLocal);

      uint64_t s1 = rdtsc_serialized();
      double stepSecLocal = tS.seconds();

      if (n > cfg.warmupSteps) {
        ++timed;
        stepSecSum += stepSecLocal;
        poissonSecSum += poissonSecLocal;
        stepCycSum += (s1 - s0);
        poissonCycSum += poissonCycLocal;

        itSum += pst.iters;
        resSum += pst.res_inf;
        dltSum += pst.max_delta;
      }

      if (n % 50 == 0) {
        std::cerr << "step " << n
                  << "  max|div|=" << max_divergence()
                  << "  poisson iters=" << pst.iters
                  << "  res_inf=" << pst.res_inf
                  << "  maxDelta=" << pst.max_delta
                  << "\n";
      }

      if (cfg.writeVtk && (n % cfg.vtkEvery == 0)) {
        write_vtk(frame++);
      }
    }

    st.maxDiv = max_divergence();

    if (timed > 0) {
      st.avgStepSec = stepSecSum / timed;
      st.avgPoissonSec = poissonSecSum / timed;
      st.avgStepCycles = (double)stepCycSum / (double)timed;
      st.avgPoissonCycles = (double)poissonCycSum / (double)timed;

      st.avgPoissonIters = itSum / timed;
      st.avgPoissonResInf = resSum / timed;
      st.avgPoissonMaxDelta = dltSum / timed;
    }

    if (cfg.writeCenterline) {
      write_centerlines_csv();
      std::cerr << "Wrote centerlines: "
                << cfg.csvDir << "/" << cfg.prefix << "_u_x0p5.csv and "
                << cfg.csvDir << "/" << cfg.prefix << "_v_y0p5.csv\n";
    }
    if (cfg.writeGhia) {
      write_ghia_csv();
      std::cerr << "Wrote Ghia samples: "
                << cfg.csvDir << "/" << cfg.prefix << "_ghia_u.csv and "
                << cfg.csvDir << "/" << cfg.prefix << "_ghia_v.csv\n";
    }

    return st;
  }
};

// ------------------------------ CLI ------------------------------
static inline std::vector<int> parse_csv_ints(const std::string &s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
  }
  return out;
}

static void help() {
  std::cout <<
R"(Usage:
  ./cavity [options]

Core:
  --solver jacobi|sor     Poisson solver (default sor)
  --Nx N --Ny N           grid size (default 128x128)
  --Re R                  Reynolds number (default 400)
  --steps N               timesteps (default 5000)

Poisson:
  --tol T                 residual inf-norm tolerance (default 1e-6)
  --maxIters N            max Poisson iterations per step (default 8000)
  --checkEvery K          compute residual every K iters (default 25)  [speed]
  --deltaTol D            optional early stop on max update (default 0=off)
  --omega W               SOR relaxation (default 1.7) [SOR only]

Time:
  --dtMax dt              max dt (default 0.01)
  --cfl c                 CFL target (default 0.5)
  --fixedDt dt            use fixed dt instead of CFL/diffusion dt

Output:
  --noVtk                 disable VTK output (recommended for timing)
  --vtkEvery K            write VTK every K steps (default 200)
  --vtkDir DIR            folder for VTK output (default vtk_out)
  --csvDir DIR            folder for CSV output (default csv_out)
  --centerline            write centerline CSVs at end
  --ghia                  write Ghia sample-point CSVs at end
  --prefix NAME           prefix for CSV outputs (default "run")

Timing:
  --warmup N              warmup steps excluded from averages (default 200)

Experiments:
  --sweep a,b,c           run Nx=Ny in {a,b,c} and print CSV summary table

Examples:
  ./cavity --solver sor --Nx 128 --Ny 128 --Re 400 --steps 800 --warmup 200 --noVtk --tol 1e-4 --maxIters 2000 --checkEvery 25 --omega 1.9
  ./cavity --solver jacobi --Nx 64 --Ny 64 --Re 100 --steps 1500 --warmup 200 --noVtk --tol 1e-4 --maxIters 4000 --checkEvery 50
  ./cavity --sweep 32,64,128 --solver sor --Re 400 --steps 800 --warmup 200 --noVtk --tol 1e-4 --maxIters 2000 --checkEvery 25
)";
}

int main(int argc, char **argv) {
  SimConfig cfg;
  std::string solverName = "sor";
  std::string sweepList;

  auto need = [&](int &i, const std::string &flag) -> std::string {
    if (i + 1 >= argc) {
      std::cerr << "Missing value after " << flag << "\n";
      std::exit(1);
    }
    return std::string(argv[++i]);
  };

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") { help(); return 0; }
    else if (arg == "--solver") solverName = need(i, arg);
    else if (arg == "--Nx") cfg.Nx = std::atoi(need(i, arg).c_str());
    else if (arg == "--Ny") cfg.Ny = std::atoi(need(i, arg).c_str());
    else if (arg == "--Re") cfg.Re = std::atof(need(i, arg).c_str());
    else if (arg == "--steps") cfg.steps = std::atoi(need(i, arg).c_str());

    else if (arg == "--tol") cfg.poissonTol = std::atof(need(i, arg).c_str());
    else if (arg == "--maxIters") cfg.poissonMaxIters = std::atoi(need(i, arg).c_str());
    else if (arg == "--checkEvery") cfg.poissonCheckEvery = std::atoi(need(i, arg).c_str());
    else if (arg == "--deltaTol") cfg.poissonDeltaTol = std::atof(need(i, arg).c_str());
    else if (arg == "--omega") cfg.sorOmega = std::atof(need(i, arg).c_str());

    else if (arg == "--dtMax") cfg.dtMax = std::atof(need(i, arg).c_str());
    else if (arg == "--cfl") cfg.cfl = std::atof(need(i, arg).c_str());
    else if (arg == "--fixedDt") { cfg.fixedDt = true; cfg.dtFixed = std::atof(need(i, arg).c_str()); }

    else if (arg == "--noVtk") cfg.writeVtk = false;
    else if (arg == "--vtkEvery") cfg.vtkEvery = std::atoi(need(i, arg).c_str());
    else if (arg == "--vtkDir") cfg.vtkDir = need(i, arg);
    else if (arg == "--csvDir") cfg.csvDir = need(i, arg);

    else if (arg == "--centerline") cfg.writeCenterline = true;
    else if (arg == "--ghia") cfg.writeGhia = true;
    else if (arg == "--prefix") cfg.prefix = need(i, arg);

    else if (arg == "--warmup") cfg.warmupSteps = std::atoi(need(i, arg).c_str());

    else if (arg == "--sweep") sweepList = need(i, arg);

    else {
      std::cerr << "Unknown option: " << arg << "\n";
      help();
      return 1;
    }
  }

  auto run_one = [&](int N) {
    cfg.Nx = N;
    cfg.Ny = N;

    PoissonJacobi jacobi;
    PoissonSOR sor(cfg.sorOmega);
    PoissonSolver *ps = nullptr;

    if (solverName == "jacobi") ps = &jacobi;
    else if (solverName == "sor") ps = &sor;
    else {
      std::cerr << "Unknown solver: " << solverName << " (use jacobi or sor)\n";
      std::exit(1);
    }

    std::cerr << "\n=== Run: N=" << N
              << " Re=" << cfg.Re
              << " solver=" << ps->name()
              << " tol=" << cfg.poissonTol
              << " maxIters=" << cfg.poissonMaxIters
              << " checkEvery=" << cfg.poissonCheckEvery
              << " deltaTol=" << cfg.poissonDeltaTol
              << " omega=" << cfg.sorOmega
              << " steps=" << cfg.steps
              << " warmup=" << cfg.warmupSteps
              << " VTK=" << (cfg.writeVtk ? "on" : "off")
              << " vtkDir=" << cfg.vtkDir
              << " csvDir=" << cfg.csvDir
              << " ===\n";

    // For sweep: make prefixes unique if writing CSVs
    SimConfig local = cfg;
    if (!sweepList.empty() && (local.writeCenterline || local.writeGhia)) {
      std::ostringstream p;
      p << local.prefix << "_N" << N << "_" << ps->name();
      local.prefix = p.str();
    }

    Simulation sim(local, ps);
    RunStats st = sim.run();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "SUMMARY"
              << " N=" << N
              << " Re=" << local.Re
              << " solver=" << ps->name()
              << " avgStepSec=" << st.avgStepSec
              << " avgPoissonSec=" << st.avgPoissonSec
              << " avgStepCycles=" << st.avgStepCycles
              << " avgPoissonCycles=" << st.avgPoissonCycles
              << " avgPoissonIters=" << st.avgPoissonIters
              << " avgPoissonResInf=" << st.avgPoissonResInf
              << " avgPoissonMaxDelta=" << st.avgPoissonMaxDelta
              << " maxDiv=" << st.maxDiv
              << "\n";

    return st;
  };

  if (!sweepList.empty()) {
    auto Ns = parse_csv_ints(sweepList);
    if (Ns.empty()) {
      std::cerr << "Empty sweep list.\n";
      return 1;
    }

    std::cout << "N,Re,solver,avgStepSec,avgPoissonSec,avgStepCycles,avgPoissonCycles,avgPoissonIters,avgPoissonResInf,avgPoissonMaxDelta,maxDiv\n";
    for (int N : Ns) {
      RunStats st = run_one(N);
      std::cout << N << "," << cfg.Re << "," << solverName << ","
                << st.avgStepSec << "," << st.avgPoissonSec << ","
                << st.avgStepCycles << "," << st.avgPoissonCycles << ","
                << st.avgPoissonIters << "," << st.avgPoissonResInf << ","
                << st.avgPoissonMaxDelta << "," << st.maxDiv
                << "\n";
    }
    return 0;
  }

  run_one(cfg.Nx);
  return 0;
}

