#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/; cd ../.. && pwd

echo 'Experiment 1a ---> Offload training Basic Agent'
#python -m src.run
echo '  Launching server...'
python -m src.run_server &
sleep 3

for cid in `seq 1 2`; do
  python src/run_client.py --cid=${cid} &
done


#echo 'Experiment 1b ---> Offload training RL Agent'

#echo 'Experiment 2  ---> Reproduce RL Agent training'
#echo 'Experiment 3  ---> Clustered FL train'

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
