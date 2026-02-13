set -euo pipefail

./OP --solver sor --Nx 32 --Ny 32 --Re 400 --steps 1000 --warmup 200 --noVtk --tol 1e-4 --maxIters 2000 --checkEvery 25 --omega 1.9
