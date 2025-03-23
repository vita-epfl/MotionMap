trap 'trap " " SIGTERM; kill 0; wait' SIGTERM
python -u main.py &
wait