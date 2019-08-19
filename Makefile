python_cmd = python
virtualenv_cmd := $(python_cmd) -m virtualenv
pip_cmd := $(python_cmd) -m pip

clean:
	rm -rf venv
	rm -rf venv-dev
	rm -rf dist
	rm -rf build
	rm -rf qubovert.egg-info

install_dev:
	$(virtualenv_cmd) venv-dev
	source venv-dev/bin/activate
	$(pip_cmd) install -r requirements-dev.txt
	$(pip_cmd) install -e .

install:
	$(virtualenv_cmd) venv
	source venv/bin/activate
	$(pip_cmd) install -r requirements.txt
	$(pip_cmd) install -e .

test:
	$(python_cmd) -m pydocstyle convention=numpy qubovert
	$(python_cmd) -m pytest --codestyle
	$(python_cmd) setup.py sdist bdist_wheel
	$(python_cmd) -m twine check dist/*

run_tests:
	install_dev
	test
	clean
