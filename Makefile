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
	. venv-dev/bin/activate && $(pip_cmd) install -r requirements-dev.txt
	. venv-dev/bin/activate && $(pip_cmd) install -e .

install:
	$(virtualenv_cmd) venv
	. venv/bin/activate && $(pip_cmd) install -r requirements.txt
	. venv/bin/activate && $(pip_cmd) install -e .

test_dev:
	env -i bash -c "source venv-dev/bin/activate && python -m pydocstyle convention=numpy qubovert && python -m pytest --codestyle && python setup.py sdist bdist_wheel && python -m twine check dist/*"

test:
	env -i bash -c "source venv/bin/activate && python -m pydocstyle convention=numpy qubovert && python -m pytest --codestyle && python setup.py sdist bdist_wheel && python -m twine check dist/*"
