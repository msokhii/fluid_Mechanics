set -euo pipefail

g++ -O3 -march=native -std=c++17 -DNDEBUG solver.cpp -o OP
