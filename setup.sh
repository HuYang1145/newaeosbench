set -e

project_root=$(dirname $(realpath $0))

curl https://raw.githubusercontent.com/LutingWang/todd/main/bin/pipenv_install | bash -s -- 3.11.10

pipenv run pip install /archive/wheels/torch-2.6.0+cu124-cp311-cp311-linux_x86_64.whl
pipenv run pip install -i https://download.pytorch.org/whl/cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124

pipenv run pip install git+https://github.com/lvis-dataset/lvis-api.git@lvis_challenge_2021 --no-build-isolation
make install_todd

pipenv run pip install \
    regex \
    "cmake<4.0" \
    "conan<2.0" \
    wheel \
    gymnasium \
    "stable-baselines3[extra]" \
    pymap3d

cd third_party/basilisk
sudo apt install swig
# Checkout to version 786cb285d (last version before numpy 2.0 support)c3624e0fe9d79c2be705b1c587ba88e34f8061ff
# This version is compatible with numpy 1.x required by todd-ai
git checkout 786cb285d
pip install pytest-html
CMAKE_TLS_VERIFY=0 python conanfile.py
