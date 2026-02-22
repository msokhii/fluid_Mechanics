#!/usr/bin/env bash
set -euo pipefail

# =========================
# User settings (edit these)
# =========================
EXE="./cavity"
SRC="solver.cpp"

RE=400
STEPS=3500
WARMUP=400

TOL="1e-4"
MAXITERS=5000
CHECKEVERY=15
OMEGA=1.9

# Grid sizes to compare
N_LIST=(32 64 128)

# Toggle runs
DO_TIMING_SWEEP=1      # no VTK, for fair timing comparison
DO_REPORT_RUNS=1       # writes VTK + CSV for selected cases below

# For report outputs (VTK/CSV), choose which cases to generate
REPORT_SOLVERS=("jacobi" "sor")    # e.g. ("sor") or ("jacobi" "sor")
REPORT_N_LIST=(32 64 128)       # e.g. (64 128)
VTK_EVERY=100

# Root output folder
STAMP=$(date +"%Y%m%d_%H%M%S")
ROOT="results_${STAMP}"

# =========================
# Build
# =========================
echo "[build] Compiling ${SRC} -> ${EXE}"
g++ -O3 -march=native -std=c++17 -DNDEBUG "${SRC}" -o "${EXE}"

mkdir -p "${ROOT}"

# =========================
# Helpers
# =========================
summary_csv="${ROOT}/summary.csv"
echo "N,Re,solver,avgStepSec,avgPoissonSec,avgStepCycles,avgPoissonCycles,avgPoissonIters,avgPoissonResInf,avgPoissonMaxDelta,maxDiv" > "${summary_csv}"

parse_summary_to_csv() {
  # Input: one SUMMARY line
  # Output: CSV row (same column order as header above)
  local line="$1"
  awk '
  {
    for (i = 1; i <= NF; ++i) {
      split($i, a, "=")
      if (length(a[2]) > 0) kv[a[1]] = a[2]
    }
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
      kv["N"], kv["Re"], kv["solver"],
      kv["avgStepSec"], kv["avgPoissonSec"],
      kv["avgStepCycles"], kv["avgPoissonCycles"],
      kv["avgPoissonIters"], kv["avgPoissonResInf"],
      kv["avgPoissonMaxDelta"], kv["maxDiv"]
  }' <<< "${line}"
}

run_case() {
  local solver="$1"
  local N="$2"
  local mode="$3"   # "timing" or "report"

  local case_dir="${ROOT}/${mode}/${solver}_N${N}"
  local vtk_dir="${case_dir}/vtk"
  local csv_dir="${case_dir}/csv"
  local prefix="N${N}_${solver}"

  mkdir -p "${case_dir}"

  local stdout_file="${case_dir}/stdout.txt"
  local stderr_file="${case_dir}/run.log"
  local cmd=(
    "${EXE}"
    --solver "${solver}"
    --Nx "${N}" --Ny "${N}"
    --Re "${RE}"
    --steps "${STEPS}"
    --warmup "${WARMUP}"
    --tol "${TOL}"
    --maxIters "${MAXITERS}"
    --checkEvery "${CHECKEVERY}"
    --omega "${OMEGA}"
  )

  if [[ "${mode}" == "timing" ]]; then
    # Fair timing: disable file output
    cmd+=(--noVtk)
  else
    # Report mode: write VTK + CSV outputs into dedicated folders
    cmd+=(--vtkEvery "${VTK_EVERY}")
    cmd+=(--centerline --ghia)
    cmd+=(--vtkDir "${vtk_dir}" --csvDir "${csv_dir}" --prefix "${prefix}")
  fi

  echo
  echo "[run] mode=${mode} solver=${solver} N=${N}"
  echo "      output: ${case_dir}"
  echo "      cmd: ${cmd[*]}"

  # stdout contains SUMMARY line, stderr contains step progress
  "${cmd[@]}" > "${stdout_file}" 2> "${stderr_file}"

  # Extract and store summary
  local summary_line
  summary_line=$(grep '^SUMMARY' "${stdout_file}" | tail -n 1 || true)

  if [[ -z "${summary_line}" ]]; then
    echo "[warn] No SUMMARY line found for ${solver} N=${N}. Check ${stderr_file}"
    return 1
  fi

  echo "${summary_line}" | tee -a "${ROOT}/summary_all.txt" >/dev/null
  parse_summary_to_csv "${summary_line}" >> "${summary_csv}"
}

# =========================
# 1) Timing sweep (no VTK)
# =========================
if [[ "${DO_TIMING_SWEEP}" -eq 1 ]]; then
  for solver in jacobi sor; do
    for N in "${N_LIST[@]}"; do
      run_case "${solver}" "${N}" "timing"
    done
  done
fi

# =========================
# 2) Report runs (VTK + CSV)
# =========================
if [[ "${DO_REPORT_RUNS}" -eq 1 ]]; then
  for solver in "${REPORT_SOLVERS[@]}"; do
    for N in "${REPORT_N_LIST[@]}"; do
      run_case "${solver}" "${N}" "report"
    done
  done
fi

echo
echo "Done."
echo "Root folder: ${ROOT}"
echo "Timing/report summary CSV: ${summary_csv}"
echo "All SUMMARY lines: ${ROOT}/summary_all.txt"
