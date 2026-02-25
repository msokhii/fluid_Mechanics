/* 
Compile using g++ -O3 -march=native -std=c++17 -DNDEBUG solver.cpp -o OP

For timings compile:
./OP --solver _NAME_ --Nx _SIZE_ --Ny _SIZE_ --Re _DEF_ --timeStep _DEF_ --warmup _DEF_ --noVtk \
--tol _DEF_ --maxIters _DEF_ --checkEvery _DEF_ --omega _DEF_

For VTK OP's compil:
./OP --solver _NAME_ --Nx _SIZE_ --Ny _SIZE_ --Re _DEF_ --timeStep _DEF_ --vtkEvery _DEF_ \
--centerline --ghia --prefix NDEF_NAME
*/

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
using LONG=int64_t;
using ULNG=uint_fast64_t;
namespace fs=filesystem; //Alias.

/*
Checks for CPU clock cycles for the entire time step(Poisson solve) 
only if we are in x86-64, otherwise returns 0 (No cycle counts).
*/

#if defined(__x86_64__)||defined(_M_X64)
#include<x86intrin.h>
static inline ULNG rdtsc_serialized(){
    unsigned aux=0;
    return __rdtscp(&aux);
  }
#else
  static inline ULNG rdtsc_serialized(){return 0;}
#endif

/*
Grid builder. We are using 32^2,64^2 and 128^2.
*/

struct Field{
    /*
    This is a 2D array-grid stored in a 1D vector.
    */
    int nx=0;
    int ny=0;
    vector<double>a; 
    /*
    Constructors.
    */
    Field()=default;
    Field(int nx_,int ny_,double val=0.0){ 
        resize(nx_,ny_,val);
    }
    /*
    Suppose, nx=5 and ny=4. This is a 5x4 grid which is stored in 
    a 1D vector a i.e. 20 values all initialized to 0.
    */
    void resize(int nx_,int ny_,double val=0.0) {
        nx=nx_;
        ny=ny_;
        a.assign((int)nx*(int)ny,val);
        /*
        Check:
        size(a); -> Should be 20 based on above example.
        */
    }
    /*
    Indexing formula used here is: (i+nx*j). For example, suppose we 
    want to store q[3,2] and nx=5. This would be stored in a[13].
    */
    // This version is read and write.
    inline double &operator()(int i,int j){
        return a[(int)i+(int)nx*(int)j];
    }
    // This version is read only.
    inline double operator()(int i,int j)const{
        return a[(int)i+(int)nx*(int)j];
    }
    /*
    Initialized all entires of a to that of val.
    */
    void fillVec(double val){ 
        fill(a.begin(),a.end(),val); 
    }
};

/*
Basic timing routine.
*/

struct Timer{
    using clock=chrono::steady_clock;
    clock::time_point t0;
    void start(){ 
        t0=clock::now(); 
    }
    double seconds()const{
        return chrono::duration<double>(clock::now()-t0).count(); //Returning elapsed seconds after t0.
    }
};

/*
Helper functions for solving the pressure poisson equation i.e. lap(phi)=RHS where
BC's are Neumann.
Basically, given RHS(i,j) everywhere we want to find phi(i,j) such that the 
discrete laplacian of phi equals those values.
*/

static inline void applyNeumannBC(Field &q,int Nx,int Ny){
  /*
  We are enforcing ghost cells on left,right,top and bottom. Note, 
  actual grid size in this case is (Nx+2)*(Ny+2). So,
  i=0,i=Nx+1,j=0,j=Ny+1 are ghost. Doing this makes the one sided difference in 
  boundary 0 i.e partial(q)/partial(x,y)=0.
  */
  for(int j=1;j<=Ny;j++){
      /*
      Copying the nearest value to the ghost cells to apply Neumann BC's with ease.
      */
      q(0,j)=q(1, j);
      q(Nx+1,j)=q(Nx,j);
  }
  for(int i=1;i<=Nx;i++){
      q(i,0)=q(i,1);
      q(i,Ny+1)=q(i,Ny);
  }
  //Corner values.
  q(0,0)=q(1,1);
  q(0,Ny+1)=q(1,Ny);
  q(Nx+1,0)=q(Nx,1);
  q(Nx+1,Ny+1)=q(Nx,Ny);
}

/*
We want a unique solution. Recall for a Neumann poisson equation, if phi is a solution
then so is phi+C. We impose uniqueness by requiring Mean(phi)=0.
*/

static inline void imposeUNQ(Field &q,int Nx,int Ny){
  double sum=0.0;
  /*
  Sums the interior values -> Computes average -> Subtracts this from interior values.
  This is done in place.
  */
  for(int j=1;j<=Ny;j++){
      for(int i=1;i<=Nx;i++){
          sum+=q(i,j);
      }
  }  
  double mean=sum/(double)(Nx*Ny);
  for(int j=1;j<=Ny;j++){
      for(int i=1;i<=Nx;i++){
          q(i,j)-=mean;
      }
  }
}
/*
Following function OP is what we are using to test for convergence with Ghia. 
This computes the inf norm of the residual of the poisson equation i.e. it does 
r[i,j]=b[i,j]-lap(x[i,j]) and OP's max(abs(r[i,j])).

LAP is the 5 point stencil we are using. Explicitly: 

lap(x)=(x[i+1,j]-2*x[i-1,j])/dx^2+(x[i,j+1]-2*x[i,j]+x[i,j-1])/dy^2.
*/

static inline double compINFNORM(const Field &x,const Field &b,
                                int Nx,int Ny,double dx,double dy){

  // Precomputation for speed.
    const double invdx2=1.0/(dx*dx);
    const double invdy2=1.0/(dy*dy);
    double rinf=0.0;
    // We do this for each interior cells (avoid the ghost cells).
    for(int j=1;j<=Ny;j++){
        for(int i=1;i<=Nx;i++){
            double lap=(x(i+1,j)-2.0*x(i,j)+x(i-1,j))*invdx2
                       +(x(i,j+1)-2.0*x(i,j)+x(i,j-1))*invdy2;
            double r=b(i,j)-lap;
            rinf=max(rinf,abs(r));
        }
    }
    return rinf;
}

/*
Keeping track of number of iterations for each poisson solve.
*/

struct pSolverINFO{
    int iters=0;
    double res_inf=0.0;
};

/*
Basic interface for JacobiSolve and SORSolve. Both methods make use of this struct.
*/

struct pSOLVER{
    virtual ~pSOLVER() = default;
    virtual string name() const = 0;
    // API Call.
    virtual pSolverINFO solve(Field &phi, const Field &rhs,
                              int Nx, int Ny, double dx, double dy,
                              int maxIters, double tol,
                              int checkEvery) = 0;
};

/*
My implementation for the Jacobi method to solve the pressure poisson equation.
*/

struct jacobiSOLVER final:public pSOLVER{
    string name() const override{ 
        return "jacobi"; 
    }
    pSolverINFO solve(Field &phi,const Field &rhs,
                     int Nx,int Ny,double dx,double dy,
                     int maxIters,double tol,
                     int checkEvery) override{
        Field next(phi.nx,phi.ny,0.0);
        // Precomputation for speed.
        /*
        Note, we are solving: 
        phi[i,j]=((phi_E+phi_W)/dx^2+(phi_N+phi_S)/dy^2-RHS[i,j])/2*(1/dx^2+1/dy^2). This is 
        the denom below.
        */
        const double invdx2=1.0/(dx*dx);
        const double invdy2=1.0/(dy*dy);
        const double denom=2.0*(invdx2+invdy2);
        applyNeumannBC(phi, Nx, Ny);
        imposeUNQ(phi, Nx, Ny);
        double rinf=compINFNORM(phi,rhs,Nx,Ny,dx,dy);
        double maxDelta=0.0;
        int it=0;
        for(int k=1;k<=maxIters;k++){
            /*
            During each iteration, we impose Neumann BC's on the current PHI. 
            */
            it=k;
            applyNeumannBC(phi,Nx,Ny);
            // Jacobi update formula referenced.
            for(int j=1;j<=Ny;j++){
                for(int i=1;i<=Nx;i++){
                    next(i,j)=((phi(i+1,j)+phi(i-1,j))*invdx2
                    +(phi(i,j+1)+phi(i,j-1))*invdy2-rhs(i,j))/denom;
                }
            }
            // Now, we do phi=next and measure the largest change.
            maxDelta = 0.0;
            for(int j=1;j<=Ny;j++){
                for(int i=1;i<=Nx;i++){
                    double d=abs(next(i,j)-phi(i,j));
                    maxDelta=max(maxDelta,d);
                    phi(i,j)=next(i,j);
                }
            }
            imposeUNQ(phi,Nx,Ny);
            applyNeumannBC(phi,Nx,Ny);
            /*
            This is a bottleneck so only do it after x iterations.
            */
            if(checkEvery>0 &&(it%checkEvery==0)){
                rinf=compINFNORM(phi,rhs,Nx,Ny,dx,dy);
                if(rinf<tol){
                    break;
                }
            }
            // Data purposes.
            rinf=compINFNORM(phi,rhs,Nx,Ny,dx,dy);
            applyNeumannBC(phi,Nx,Ny);
            return pSolverINFO{it,rinf};
        }
    }
};

/*
My implementation for the SOR method to solve the pressure poisson equation.
*/

// IGNORING THIS FOR NOW.

struct PoissonSOR final : public pSOLVER {
  double omega = 1.7;
  explicit PoissonSOR(double w) : omega(w) {}
  std::string name() const override { return "sor"; }

  pSolverINFO solve(Field &phi, const Field &rhs,
                     int Nx, int Ny, double dx, double dy,
                     int maxIters, double tol,
                     int checkEvery) override {
    const double invdx2 = 1.0 / (dx * dx);
    const double invdy2 = 1.0 / (dy * dy);
    const double denom  = 2.0 * (invdx2 + invdy2);

    applyNeumannBC(phi, Nx, Ny);
    imposeUNQ(phi, Nx, Ny);

    double rinf = compINFNORM(phi, rhs, Nx, Ny, dx, dy);
    double maxDelta = 0.0;
    int it = 0;

    for (int k = 1; k <= maxIters; ++k) {
      it = k;
      applyNeumannBC(phi, Nx, Ny);

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

      imposeUNQ(phi, Nx, Ny);
      applyNeumannBC(phi, Nx, Ny);

      if (checkEvery > 0 && (it % checkEvery == 0)) {
        rinf = compINFNORM(phi, rhs, Nx, Ny, dx, dy);
        if (rinf < tol) break;
      }
    }

    rinf = compINFNORM(phi, rhs, Nx, Ny, dx, dy);
    applyNeumannBC(phi, Nx, Ny);

    return pSolverINFO{it, rinf};
  }
};

/*
My main struct to run all simulations.
*/

struct runSim{
  int Nx=128; 
  int Ny=128;
  double Lx=1.0;
  double Ly=1.0;
  double U_lid=1.0;
  double Re=800.0;

  // OP data directory.
  string vtkDir="vtkDATA";
  string csvDir="csvDATA";

  int timeStep=6500;
  int vtkEvery=50;
  bool writeVtk=true;

  // Default parameters for timeStep.
  double dtMax=0.01;
  double CFL=0.5;
  bool fixedDt=false;
  double dtFixed=0.005;

  // Default poisson solver settings.
  int poissonMaxIters=10000;
  double poissonTol=1e-6; 
  int poissonCheckEvery=25;
  double sorOmega = 1.7; // relaxation paramater for when running SOR.
  int warmupTS=100; // This is not included when timing routines.

  // File stuff.
  bool writeCenterline=false;
  bool writeGhia=false;
  string prefix="RUN";
};

/*
Struct for generating report.
*/

struct RunStats {
  // Initializations for average values.
  double avgtimeStepec=0.0;
  double avgPoissonSec=0.0;
  double avgStepCycles=0.0;
  double avgPoissonCycles=0.0;
  double avgPoissonIters=0.0;
  double avgPoissonResInf=0.0;
  double avgPoissonMaxDelta=0.0;
  double maxDiv=0.0;
};

struct Simulation {
  runSim simOBJ;
  double dx;
  double dy;
  double nu; // This is computed from using the RE formula from lec. notes.
  Field p;
  Field phi;
  Field rhs;
  // Same as above but this is for a MAC style grid instead.
  Field u;
  Field v;
  Field us;
  Field vs;
  pSOLVER *ps=nullptr;
  // Default constructor.
  explicit Simulation(const runSim&c,pSOLVER *solver):simOBJ(c),ps(solver){
    dx=simOBJ.Lx/simOBJ.Nx;
    dy=simOBJ.Ly/simOBJ.Ny;
    nu=(simOBJ.U_lid*simOBJ.Lx)/simOBJ.Re;
    // Resizing including the ghost cells here.
    p.resize(simOBJ.Nx+2,simOBJ.Ny+2,0.0);
    phi.resize(simOBJ.Nx+2,simOBJ.Ny+2,0.0);
    rhs.resize(simOBJ.Nx+2,simOBJ.Ny+2,0.0);
    u.resize(simOBJ.Nx+1,simOBJ.Ny+2,0.0);
    v.resize(simOBJ.Nx+2,simOBJ.Ny+1,0.0);
    us.resize(simOBJ.Nx+1,simOBJ.Ny+2,0.0);
    vs.resize(simOBJ.Nx+2,simOBJ.Ny+1,0.0);
    // Folder stuff.
    if (simOBJ.writeVtk){
      fs::create_directories(simOBJ.vtkDir);
    }
    if(simOBJ.writeCenterline||simOBJ.writeGhia){
      fs::create_directories(simOBJ.csvDir);
    }
  }

  /*
  Note, we are assuming that the left and right walls are no slip i.e.
  water sticks to the walls.
  */
  void applyVBC(Field &uu,Field &vv){
    const int Nx=simOBJ.Nx;
    const int Ny=simOBJ.Ny;
    /*
    Left wall and right wall u=0.
    */
    for(int j=1;j<=Ny;j++){
      uu(0,j)=0.0;
      uu(Nx,j)=0.0;
    }
    /*
    Wall velocity at the boundary midpoint=0. I am not sure if this works?
    */
    for(int i=0;i<=Nx;i++){
      uu(i,0)=-uu(i,1);
    }
    /*
    This is for the top moving lid.
    */
    for(int i=0;i<=Nx;i++){
      uu(i,Ny+1)=2.0*simOBJ.U_lid-uu(i, Ny);
    }
    // BC for v.
    for(int i=1;i<=Nx;i++){
      vv(i,0)=0.0;
      vv(i,Ny)=0.0;
    }
    for(int j=0;j<=Ny;j++){
      vv(0,j)=-vv(1,j);
      vv(Nx+1,j)=-vv(Nx,j);
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
    if (simOBJ.fixedDt) return simOBJ.dtFixed;

    double umax = 0.0;
    for (int j = 1; j <= simOBJ.Ny; ++j)
      for (int i = 0; i <= simOBJ.Nx; ++i)
        umax = std::max(umax, std::abs(u(i, j)));
    for (int j = 0; j <= simOBJ.Ny; ++j)
      for (int i = 1; i <= simOBJ.Nx; ++i)
        umax = std::max(umax, std::abs(v(i, j)));

    double hmin = std::min(dx, dy);
    double dt_adv  = (umax > 1e-12) ? simOBJ.CFL * hmin / umax : simOBJ.dtMax;
    double dt_diff = (nu > 0.0) ? 0.25 * (hmin * hmin) / nu : simOBJ.dtMax;
    return std::min(simOBJ.dtMax, std::min(dt_adv, dt_diff));
  }

  double max_divergence() const {
    double md = 0.0;
    for (int j = 1; j <= simOBJ.Ny; ++j) {
      for (int i = 1; i <= simOBJ.Nx; ++i) {
        double div = (u(i, j) - u(i-1, j)) / dx
                   + (v(i, j) - v(i, j-1)) / dy;
        md = std::max(md, std::abs(div));
      }
    }
    return md;
  }

  /*
  PARAVIEW COMMANDS:
  */

  void opVTK(int frame)const{
      if(!simOBJ.writeVtk){return;}
      fs::create_directories(simOBJ.vtkDir);
      ostringstream fn;
      fn<<simOBJ.vtkDir<<"/out_"<<std::setw(6)<<std::setfill('0')<<frame<<".vtk";
      ofstream out(fn.str());
      if(!out){return;}
      /*
      Recall, data is in the center of the cell with origin (dx/2,dy/2).
      */
      out<<"VTK DataFile\n";
      out<<"ASCII\n";
      out<<"DATASET_POINTS\n";
      out<<"DIMENSIONS "<<simOBJ.Nx<<" "<<simOBJ.Ny<<" 1\n";
      out<<"ORIGIN "<<0.5*dx<<" "<<0.5*dy<<" 0\n";
      out<<"SPACING "<<dx<<" "<<dy<<" 1\n";
      out<<"POINT_DATA "<<(simOBJ.Nx*simOBJ.Ny)<<"\n";
      // This is cell centered pressure.
      out<<"SCALARS pressure double 1\n";
      out<<"LOOKUP_TABLE default\n";
      for(int j=1;j<=simOBJ.Ny;j++){
          for(int i=1;i<=simOBJ.Nx;i++){
              out<<p(i,j)<<"\n";
          }
      }
      /*
      This is explicitly for visualization in PARAVIEW. We are converting staggered 
      MAC velocities to cell centered velocities by averaging neighbouring grid values.
      */
      out<<"VECTORS velocity double\n";
      for(int j=1;j<=simOBJ.Ny;j++){
          for(int i=1;i<=simOBJ.Nx;i++){
              double uc=0.5*(u(i,j)+u(i-1,j));
              double vc=0.5*(v(i,j)+v(i,j-1));
              out<<uc<<" "<<vc<<" 0\n";
          }
      }
  }
  // Cell centered sampling from MAC.
  inline double uc(int i,int j)const{
      return 0.5*(u(i,j)+u(i-1,j)); 
  }
  inline double vc(int i,int j)const{
      return 0.5*(v(i,j)+v(i,j-1)); 
  }

  double sample_uc(double x, double y) const {
    // bilinear on cell centers
    const int Nx = simOBJ.Nx, Ny = simOBJ.Ny;
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
    const int Nx = simOBJ.Nx, Ny = simOBJ.Ny;
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

  /*
  Exporting CSV files.
  */

  void write_centerlines_csv() const {
    fs::create_directories(simOBJ.csvDir);

    // u(x=0.5,y) on Ny points (cell centers), v(x,y=0.5) on Nx points
    {
      std::ofstream out(simOBJ.csvDir + "/" + simOBJ.prefix + "_u_xATp5.csv");
      if (out) {
        out << "y,u\n";
        for (int j = 1; j <= simOBJ.Ny; ++j) {
          double y = (j - 0.5) * dy;
          double uval = sample_uc(0.5 * simOBJ.Lx, y);
          out << std::setprecision(16) << y << "," << uval << "\n";
        }
      }
    }
    {
      std::ofstream out(simOBJ.csvDir + "/" + simOBJ.prefix + "_v_y0p5.csv");
      if (out) {
        out << "x,v\n";
        for (int i = 1; i <= simOBJ.Nx; ++i) {
          double x = (i - 0.5) * dx;
          double vval = sample_vc(x, 0.5 * simOBJ.Ly);
          out << std::setprecision(16) << x << "," << vval << "\n";
        }
      }
    }
  }

  void write_ghia_csv() const {
    fs::create_directories(simOBJ.csvDir);

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
      std::ofstream out(simOBJ.csvDir + "/" + simOBJ.prefix + "_ghia_u.csv");
      if (out) {
        out << "y,u_sim\n";
        for (double y : yPts) {
          double uval;
          if (y <= 0.0) uval = 0.0;
          else if (y >= 1.0) uval = simOBJ.U_lid;
          else uval = sample_uc(0.5 * simOBJ.Lx, y * simOBJ.Ly);
          out << std::setprecision(16) << y << "," << uval << "\n";
        }
      }
    }
    {
      std::ofstream out(simOBJ.csvDir + "/" + simOBJ.prefix + "_ghia_v.csv");
      if (out) {
        out << "x,v_sim\n";
        for (double x : xPts) {
          double vval;
          if (x <= 0.0 || x >= 1.0) vval = 0.0;
          else vval = sample_vc(x * simOBJ.Lx, 0.5 * simOBJ.Ly);
          out << std::setprecision(16) << x << "," << vval << "\n";
        }
      }
    }
  }

  // One step. Returns pSolverINFO; also reports Poisson timing/cycles.
  pSolverINFO step_once(double &dt,
                         double &poissonSec, uint64_t &poissonCycles) {
    applyVBC(u, v);
    dt = compute_dt();

    // ---- Predict u* ----
    for (int j = 1; j <= simOBJ.Ny; ++j) {
      for (int i = 1; i <= simOBJ.Nx - 1; ++i) {
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
    for (int j = 1; j <= simOBJ.Ny; ++j) {
      us(0, j) = u(0, j);
      us(simOBJ.Nx, j) = u(simOBJ.Nx, j);
    }

    // ---- Predict v* ----
    for (int j = 1; j <= simOBJ.Ny - 1; ++j) {
      for (int i = 1; i <= simOBJ.Nx; ++i) {
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
    for (int i = 1; i <= simOBJ.Nx; ++i) {
      vs(i, 0) = v(i, 0);
      vs(i, simOBJ.Ny) = v(i, simOBJ.Ny);
    }
    for (int j = 0; j <= simOBJ.Ny; ++j) {
      vs(0, j) = v(0, j);
      vs(simOBJ.Nx+1, j) = v(simOBJ.Nx+1, j);
    }

    applyVBC(us, vs);

    // ---- RHS = (1/dt) div(u*,v*) ----
    rhs.fillVec(0.0);
    for (int j = 1; j <= simOBJ.Ny; ++j) {
      for (int i = 1; i <= simOBJ.Nx; ++i) {
        double div = (us(i, j) - us(i-1, j)) / dx
                   + (vs(i, j) - vs(i, j-1)) / dy;
        rhs(i, j) = div / dt;
      }
    }

    // ---- Solve Poisson ----
    phi.fillVec(0.0);
    Timer tP; tP.start();
    uint64_t c0 = rdtsc_serialized();

    pSolverINFO pst = ps->solve(
      phi, rhs, simOBJ.Nx, simOBJ.Ny, dx, dy,
      simOBJ.poissonMaxIters, simOBJ.poissonTol,
      simOBJ.poissonCheckEvery);

    uint64_t c1 = rdtsc_serialized();
    poissonCycles += (c1 - c0);
    poissonSec += tP.seconds();

    // ---- Correcting velocities ----
    for (int j = 1; j <= simOBJ.Ny; ++j) {
      for (int i = 1; i <= simOBJ.Nx - 1; ++i) {
        us(i, j) -= dt * (phi(i+1, j) - phi(i, j)) / dx;
      }
    }
    for (int j = 1; j <= simOBJ.Ny - 1; ++j) {
      for (int i = 1; i <= simOBJ.Nx; ++i) {
        vs(i, j) -= dt * (phi(i, j+1) - phi(i, j)) / dy;
      }
    }

    // pressure update
    for (int j = 1; j <= simOBJ.Ny; ++j)
      for (int i = 1; i <= simOBJ.Nx; ++i)
        p(i, j) += phi(i, j);

    applyVBC(us, vs);

    u.a.swap(us.a);
    v.a.swap(vs.a);

    return pst;
  }

  RunStats run() {
    RunStats st;
    applyVBC(u, v);

    int frame = 0;
    if (simOBJ.writeVtk) opVTK(frame++);

    // timed accumulators (exclude warmup)
    double timeStepecSum = 0.0, poissonSecSum = 0.0;
    uint64_t stepCycSum = 0, poissonCycSum = 0;
    double itSum = 0.0, resSum = 0.0, dltSum = 0.0;
    int timed = 0;

    for (int n = 1; n <= simOBJ.timeStep; ++n) {
      Timer tS; tS.start();
      uint64_t s0 = rdtsc_serialized();

      double dt = 0.0;
      double poissonSecLocal = 0.0;
      uint64_t poissonCycLocal = 0;

      pSolverINFO pst = step_once(dt, poissonSecLocal, poissonCycLocal);

      uint64_t s1 = rdtsc_serialized();
      double timeStepecLocal = tS.seconds();

      if (n > simOBJ.warmupTS) {
        ++timed;
        timeStepecSum += timeStepecLocal;
        poissonSecSum += poissonSecLocal;
        stepCycSum += (s1 - s0);
        poissonCycSum += poissonCycLocal;

        itSum += pst.iters;
        resSum += pst.res_inf;
      }

      if (n % 50 == 0) {
        std::cerr << "step " << n
                  << "  max|div|=" << max_divergence()
                  << "  poisson iters=" << pst.iters
                  << "  res_inf=" << pst.res_inf
                  << "\n";
      }

      if (simOBJ.writeVtk && (n % simOBJ.vtkEvery == 0)) {
        opVTK(frame++);
      }
    }

    st.maxDiv = max_divergence();

    if (timed > 0) {
      st.avgtimeStepec = timeStepecSum / timed;
      st.avgPoissonSec = poissonSecSum / timed;
      st.avgStepCycles = (double)stepCycSum / (double)timed;
      st.avgPoissonCycles = (double)poissonCycSum / (double)timed;

      st.avgPoissonIters = itSum / timed;
      st.avgPoissonResInf = resSum / timed;
      st.avgPoissonMaxDelta = dltSum / timed;
    }

    if (simOBJ.writeCenterline) {
      write_centerlines_csv();
      std::cerr << "WRITING CENTERLINES: "
                << simOBJ.csvDir << "/" << simOBJ.prefix << "_u_x0p5.csv and "
                << simOBJ.csvDir << "/" << simOBJ.prefix << "_v_y0p5.csv\n";
    }
    if (simOBJ.writeGhia) {
      write_ghia_csv();
      std::cerr << "WRITING SAMPLES WITH GHIA: "
                << simOBJ.csvDir << "/" << simOBJ.prefix << "_ghia_u.csv and "
                << simOBJ.csvDir << "/" << simOBJ.prefix << "_ghia_v.csv\n";
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
  --timeStep N               timetimeStep (default 5000)

Poisson:
  --tol T                 residual inf-norm tolerance (default 1e-6)
  --maxIters N            max Poisson iterations per step (default 8000)
  --checkEvery K          compute residual every K iters (default 25)  [speed]
  --deltaTol D            optional early stop on max update (default 0=off)
  --omega W               SOR relaxation (default 1.7) [SOR only]

Time:
  --dtMax dt              max dt (default 0.01)
  --CFL c                 CFL target (default 0.5)
  --fixedDt dt            use fixed dt instead of CFL/diffusion dt

Output:
  --noVtk                 disable VTK output (recommended for timing)
  --vtkEvery K            write VTK every K timeStep (default 200)
  --vtkDir DIR            folder for VTK output (default vtk_out)
  --csvDir DIR            folder for CSV output (default csv_out)
  --centerline            write centerline CSVs at end
  --ghia                  write Ghia sample-point CSVs at end
  --prefix NAME           prefix for CSV outputs (default "run")

Timing:
  --warmup N              warmup timeStep excluded from averages (default 200)

Experiments:
  --sweep a,b,c           run Nx=Ny in {a,b,c} and print CSV summary table

Examples:
  ./cavity --solver sor --Nx 128 --Ny 128 --Re 400 --timeStep 800 --warmup 200 --noVtk --tol 1e-4 --maxIters 2000 --checkEvery 25 --omega 1.9
  ./cavity --solver jacobi --Nx 64 --Ny 64 --Re 100 --timeStep 1500 --warmup 200 --noVtk --tol 1e-4 --maxIters 4000 --checkEvery 50
  ./cavity --sweep 32,64,128 --solver sor --Re 400 --timeStep 800 --warmup 200 --noVtk --tol 1e-4 --maxIters 2000 --checkEvery 25
)";
}

int main(int argc, char **argv) {
  runSim simOBJ;
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
    else if (arg == "--Nx") simOBJ.Nx = std::atoi(need(i, arg).c_str());
    else if (arg == "--Ny") simOBJ.Ny = std::atoi(need(i, arg).c_str());
    else if (arg == "--Re") simOBJ.Re = std::atof(need(i, arg).c_str());
    else if (arg == "--timeStep") simOBJ.timeStep = std::atoi(need(i, arg).c_str());

    else if (arg == "--tol") simOBJ.poissonTol = std::atof(need(i, arg).c_str());
    else if (arg == "--maxIters") simOBJ.poissonMaxIters = std::atoi(need(i, arg).c_str());
    else if (arg == "--checkEvery") simOBJ.poissonCheckEvery = std::atoi(need(i, arg).c_str());
    else if (arg == "--omega") simOBJ.sorOmega = std::atof(need(i, arg).c_str());

    else if (arg == "--dtMax") simOBJ.dtMax = std::atof(need(i, arg).c_str());
    else if (arg == "--CFL") simOBJ.CFL = std::atof(need(i, arg).c_str());
    else if (arg == "--fixedDt") { simOBJ.fixedDt = true; simOBJ.dtFixed = std::atof(need(i, arg).c_str()); }

    else if (arg == "--noVtk") simOBJ.writeVtk = false;
    else if (arg == "--vtkEvery") simOBJ.vtkEvery = std::atoi(need(i, arg).c_str());
    else if (arg == "--vtkDir") simOBJ.vtkDir = need(i, arg);
    else if (arg == "--csvDir") simOBJ.csvDir = need(i, arg);

    else if (arg == "--centerline") simOBJ.writeCenterline = true;
    else if (arg == "--ghia") simOBJ.writeGhia = true;
    else if (arg == "--prefix") simOBJ.prefix = need(i, arg);

    else if (arg == "--warmup") simOBJ.warmupTS = std::atoi(need(i, arg).c_str());

    else if (arg == "--sweep") sweepList = need(i, arg);

    else {
      std::cerr << "Unknown option: " << arg << "\n";
      help();
      return 1;
    }
  }

  auto run_one = [&](int N) {
    simOBJ.Nx = N;
    simOBJ.Ny = N;

    jacobiSOLVER jacobi;
    PoissonSOR sor(simOBJ.sorOmega);
    pSOLVER *ps = nullptr;

    if (solverName == "jacobi") ps = &jacobi;
    else if (solverName == "sor") ps = &sor;
    else {
      std::cerr << "Unknown solver: " << solverName << " (use jacobi or sor)\n";
      std::exit(1);
    }

    std::cerr << "\n=== Run: N=" << N
              << " Re=" << simOBJ.Re
              << " solver=" << ps->name()
              << " tol=" << simOBJ.poissonTol
              << " maxIters=" << simOBJ.poissonMaxIters
              << " checkEvery=" << simOBJ.poissonCheckEvery
              << " omega=" << simOBJ.sorOmega
              << " timeStep=" << simOBJ.timeStep
              << " warmup=" << simOBJ.warmupTS
              << " VTK=" << (simOBJ.writeVtk ? "on" : "off")
              << " vtkDir=" << simOBJ.vtkDir
              << " csvDir=" << simOBJ.csvDir
              << " ===\n";

    // For sweep: make prefixes unique if writing CSVs
    runSim local = simOBJ;
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
              << " avgtimeStepec=" << st.avgtimeStepec
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

    std::cout << "N,Re,solver,avgtimeStepec,avgPoissonSec,avgStepCycles,avgPoissonCycles,avgPoissonIters,avgPoissonResInf,avgPoissonMaxDelta,maxDiv\n";
    for (int N : Ns) {
      RunStats st = run_one(N);
      std::cout << N << "," << simOBJ.Re << "," << solverName << ","
                << st.avgtimeStepec << "," << st.avgPoissonSec << ","
                << st.avgStepCycles << "," << st.avgPoissonCycles << ","
                << st.avgPoissonIters << "," << st.avgPoissonResInf << ","
                << st.avgPoissonMaxDelta << "," << st.maxDiv
                << "\n";
    }
    return 0;
  }

  run_one(simOBJ.Nx);
  return 0;
}