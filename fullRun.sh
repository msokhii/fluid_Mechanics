set -euo pipefail 

./OP --solver sor --Nx 32 --Ny 32 --Re 400 --steps 5000 --vtkEvery 1000 --centerline --ghia --prefix N32_sor

