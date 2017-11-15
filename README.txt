Pierwszy raz:

Zainstaluj Pythona 3.x i PIP.
pip install virtualenv
virtualenv --no-site-packages venv
source venv/Scripts/activate
pip install -r requirements-gpu.txt

Uruchamianie zadania:
srun -p plgrid-gpu -N 1 --ntasks-per-node=1 -n 1 --gres=gpu:1 -A scqfracz --time=0:30:00 --mem-per-cpu=2GB --pty /bin/bash -l

Uruchamianie modelu:
python quality.py 100-fake
