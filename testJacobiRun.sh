set -euo piepfail 

./OP --solver jacobi --Nx 32 --Ny 32 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9 --centerline -- ghia --vtkEvery 200 --vtkDir results/vtk --csvDir results/csv --prefix N32_jacobi
