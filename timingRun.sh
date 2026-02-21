set -euo pipefail

./OP --solver sor --Nx 32 --Ny 32 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9
./OP --solver jacobi --Nx 32 --Ny 32 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9
./OP --solver sor --Nx 64 --Ny 64 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9
./OP --solver jacobi --Nx 64 --Ny 64 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9
./OP --solver sor --Nx 128 --Ny 128 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9
./OP --solver jacobi --Nx 128 --Ny 128 --Re 400 --steps 2000 --warmup 200 --noVtk --tol 1e-4 --maxIters 5000 --checkEvery 100 --omega 1.9
