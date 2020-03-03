python_cmd := python
virtualenv_cmd := $(python_cmd) -m virtualenv

clean:
	rm -rf venv
	rm -rf dist
	rm -rf build
	rm -rf qubovert.egg-info

install:
	$(python_cmd) -m pip install virtualenv
	$(virtualenv_cmd) venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -e .
	. venv/bin/activate && pip install -r requirements-dev.txt

cython_install:
	$(python_cmd) -m pip install virtualenv
	$(virtualenv_cmd) venv
	. venv/bin/activate && pip install --upgrade pip
	. venv/bin/activate && pip install -r requirements-dev.txt
	. venv/bin/activate && pip install -e .

test:
	. venv/bin/activate && python -m pydocstyle convention=numpy qubovert
	. venv/bin/activate && python -m pytest --codestyle --cov=./
	. venv/bin/activate && python setup.py sdist bdist_wheel
	. venv/bin/activate && python -m twine check dist/*

submitcoverage:
	python -m codecov
