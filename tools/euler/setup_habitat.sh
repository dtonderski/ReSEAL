python -m venv venv --system-site-packages
source venv/bin/activate
pip install cmake
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
sbatch --mem-per-cpu=8G --ntasks=4 --output build.txt --wrap="python setup.py install --headless --with-cuda --bullet"