#!/bin/sh
# Sequential GPU queue (avoids MPS contention). New this weekend.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
N=200 SUFFIX=_n200 $PY -I swarm/e4b_control.py > swarm/results/e4b_control_n200.log 2>&1
N=200 SUFFIX=_n200 $PY -I swarm/e3b_behavior.py > swarm/results/e3b_behavior_n200.log 2>&1
N=200 $PY -I swarm/e3c_monitor.py > swarm/results/e3c_monitor.log 2>&1
N=200 SUFFIX=_n200 COMPS=AAAA,DAAA,ADAA,AAAD,ADAD,DDDD,DDDA,ADDD $PY -I swarm/e4_chain.py > swarm/results/e4_chain_n200.log 2>&1
echo ALLDONE > swarm/results/queue.done
