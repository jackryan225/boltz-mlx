#!/bin/zsh
# Run one GPU job under a memory watchdog: kill it if swap use exceeds SWAP_LIMIT_MB
# (default 4000) or system memory free drops below FREE_MIN_PCT (default 15) while it runs.
# Usage: tools/guarded_run.sh <command ...>   (exit 99 = killed by the watchdog)
SWAP_LIMIT_MB=${SWAP_LIMIT_MB:-4000}
FREE_MIN_PCT=${FREE_MIN_PCT:-15}
"$@" &
pid=$!
while kill -0 $pid 2>/dev/null; do
  swap=$(sysctl -n vm.swapusage | awk '{gsub("M","",$6); print int($6)}')
  free=$(memory_pressure | tail -1 | awk '{gsub("%","",$NF); print int($NF)}')
  if [ "$swap" -gt "$SWAP_LIMIT_MB" ] || [ "$free" -lt "$FREE_MIN_PCT" ]; then
    echo "$(date '+%H:%M:%S') guarded_run: KILLED pid $pid (swap ${swap} MB > ${SWAP_LIMIT_MB} or free ${free}% < ${FREE_MIN_PCT}%): $*" >&2
    kill -TERM $pid; sleep 5; kill -KILL $pid 2>/dev/null
    exit 99
  fi
  sleep 10
done
wait $pid
