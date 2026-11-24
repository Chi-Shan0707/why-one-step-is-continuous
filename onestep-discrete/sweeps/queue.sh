#!/bin/bash
# Run the remaining sweeps after the current E3 workers finish. 2 workers per sweep.
cd "$(dirname "$0")/.."
PY=/home/chishan/miniconda3/envs/torch-gpu/bin/python
while pgrep -f "e3_sources.py --data mix16 --n_train 0" >/dev/null; do sleep 30; done
two() { for w in 0 1; do $PY "$@" --worker $w --nworkers 2 > "results/$(echo "$@" | tr ' /' '__' | cut -c1-80)_w$w.log" 2>&1 & done; wait; }
$PY sweeps/e3_budget.py > results/e3b.log 2>&1 &
two sweeps/e6_nscaling.py
wait
two sweeps/e3_sources.py --data mix16 --n_train 500
two sweeps/e3_sources.py --data mix16 --n_train 5000
two sweeps/e3_sources.py --data markov --n_train 0
echo QUEUE_DONE
