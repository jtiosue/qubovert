python_cmd := python
pip_cmd := $(python_cmd) -m pip

clean:
	$(pip_cmd) uninstall -y qubovert
	rm -rf dist
	rm -rf build
	rm -rf qubovert.egg-info

install:
	$(pip_cmd) install --upgrade pip
	$(pip_cmd) install -e .

dev_install:
	$(pip_cmd) install --upgrade pip
	$(pip_cmd) install -e .
	$(pip_cmd) install -r requirements-dev.txt

test:
	$(python_cmd) -m pydocstyle convention=numpy qubovert
	$(python_cmd) -m pytest --codestyle --cov=./ --cov-report=xml
	$(python_cmd) setup.py sdist bdist_wheel
	$(python_cmd) -m twine check dist/*

submitcoverage:
	$(python_cmd) -m codecov

upload_wheel_pypi_test:
	$(python_cmd) setup.py bdist_wheel
	$(python_cmd) -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_wheel_pypi:
	$(python_cmd) setup.py bdist_wheel
	$(python_cmd) -m twine upload dist/*

upload_source_pypi_test:
	$(python_cmd) setup.py sdist
	$(python_cmd) -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

upload_source_pypi:
	$(python_cmd) setup.py sdist
	$(python_cmd) -m twine upload dist/*

upload_manylinux_pypi_test:
	$(python_cmd) -m twine upload --repository-url https://test.pypi.org/legacy/ wheelhouse/*-manylinux*.whl

upload_manylinux_pypi:
	$(python_cmd) -m twine upload wheelhouse/*-manylinux*.whl
