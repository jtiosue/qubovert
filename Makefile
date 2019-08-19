python_cmd = python
virtualenv_cmd := $(python_cmd) -m virtualenv

clean:
    rm -rf venv
    rm -rf venv-dev
    rm -rf dist
    rm -rf build
    rm -rf qubovert.egg-info

install_dev:
    $(virtualenv_cmd) venv-dev
    source venv-dev/bin/activate
    pip install -r requirements-dev.txt
    pip install -e .

install:
    $(virtualenv_cmd) venv
    source venv/bin/activate
    pip install -r requirements.txt
    pip install -e .

test:
    python -m pydocstyle convention=numpy qubovert
    python -m pytest --codestyle
    python setup.py sdist bdist_wheel
    python -m twine check dist/*

run_tests:
    install_dev
    test
    clean
