#!/bin/sh
set -eu

uname_s="$(uname -s 2>/dev/null || echo unknown)"
cpu_flags=""

if [ "$uname_s" = "Darwin" ]; then
  cpu_flags="$(sysctl -n machdep.cpu.features 2>/dev/null || true)"
  cpu_flags="$cpu_flags $(sysctl -n machdep.cpu.leaf7_features 2>/dev/null || true)"
else
  cpu_flags="$(cat /proc/cpuinfo 2>/dev/null || true)"
fi

cpu_flags="$(printf "%s" "$cpu_flags" | tr '[:upper:]' '[:lower:]')"

if printf "%s" "$cpu_flags" | grep -q "avx512f"; then
  echo "avx512"
elif printf "%s" "$cpu_flags" | grep -q "avx2"; then
  echo "avx2"
else
  echo "generic"
fi
