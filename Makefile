python_cmd := python
virtualenv_cmd := $(python_cmd) -m venv

clean:
	rm -rf venv-dev
	rm -rf dist
	rm -rf build
	rm -rf qubovert.egg-info

install:
	$(virtualenv_cmd) venv-dev
	. venv-dev/bin/activate && pip install -r requirements-dev.txt
	. venv-dev/bin/activate && pip install .

test:
	. venv-dev/bin/activate && python -m pydocstyle convention=numpy qubovert
	. venv-dev/bin/activate && python -m pytest --codestyle --cov=./
	. venv-dev/bin/activate && python setup.py sdist bdist_wheel
	. venv-dev/bin/activate && python -m twine check dist/*

submitcoverage:
	python -m codecov
