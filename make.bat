@ECHO off

title qubovert

if "%1" == "clean" (
	deactivate
	rmdir /S /Q venv
	rmdir /S /Q dist
	rmdir /S /Q build
	rmdir /S /Q qubovert.egg-info
) else if "%1" == "deactivate" (
	deactivate
) else if "%1" == "activate" (
	venv\Scripts\activate
) else if "%1" == "dev_install" (
    python -m pip install --user virtualenv ^
    && python -m virtualenv venv ^
    && venv\Scripts\activate ^
	&& python -m pip install --upgrade pip ^
	&& pip install -e . ^
	&& pip install -r requirements-dev.txt ^
	&& echo dev_install succeeded
) else if "%1" == "install" (
    python -m pip install --user virtualenv ^
	&& python -m virtualenv venv ^
	&& venv\Scripts\activate ^
	&& python -m pip install --upgrade pip ^
	&& pip install -e . ^
	&& echo install succeeded
) else if "%1" == "cython_install" (
	rem use this if you want to use cython to recrete the c file from the pyx file
	python -m pip install --user virtualenv ^
	&& python -m virtualenv venv ^
	&& venv\Scripts\activate ^
	&& python -m pip install --upgrade pip ^
	&& pip install -r requirements-dev.txt ^
	&& pip install -e . ^
	&& echo cython_install succeeded
) else if "%1" == "test" (
	venv\Scripts\activate ^
	&& python -m pydocstyle convention=numpy qubovert ^
	&& python -m pytest --codestyle --cov=./ ^
	&& python setup.py sdist bdist_wheel ^
	&& python -m twine check dist/* ^
	&& echo tests succeeded
) else if "%1" == submitcoverage (
	venv\Scripts\activate ^
	&& python -m codecov ^
	&& echo coverage submitted
) else (
	echo Invalid option; must be either clean, dev_install, cython_install,
	echo install, test, submitcoverage, deactivate, or activate
)
