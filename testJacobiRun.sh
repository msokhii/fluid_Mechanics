set -euo pipefail 

./OP --solver jacobi --Nx 32 --Ny 32 --Re 400 --timeStep 5000 --warmup 100 --noVtk --tol 1e-4 --maxIters 10000 --checkEvery 50 --omega 1.9 --centerline -- ghia --vtkEvery 200 --vtkDir results/vtk --csvDir results/csv --prefix N32_jacobi
