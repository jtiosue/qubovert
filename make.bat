@ECHO off

pushd %~dp0

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
    python -m pip install --user virtualenv || exit /b %errorlevel%
	python -m virtualenv venv || exit /b %errorlevel%
	venv\Scripts\activate || exit /b %errorlevel%
	python -m pip install --upgrade pip || exit /b %errorlevel%
	pip install -e . || exit /b %errorlevel%
	pip install -r requirements-dev.txt || exit /b %errorlevel%
) else if "%1" == "install" (
    python -m pip install --user virtualenv || exit /b %errorlevel%
	python -m virtualenv venv || exit /b %errorlevel%
	venv\Scripts\activate || exit /b %errorlevel%
	python -m pip install --upgrade pip || exit /b %errorlevel%
	pip install -e .
) else if "%1" == "cython_install" (
	rem use this if you want to use cython to recrete the c file from the pyx file
	python -m pip install --user virtualenv || exit /b %errorlevel%
	python -m virtualenv venv || exit /b %errorlevel%
	venv\Scripts\activate || exit /b %errorlevel%
	python -m pip install --upgrade pip || exit /b %errorlevel%
	pip install -r requirements-dev.txt || exit /b %errorlevel%
	pip install -e .
) else if "%1" == "test" (
	venv\Scripts\activate
	python -m pydocstyle convention=numpy qubovert || exit /b %errorlevel%
	python -m pytest --codestyle --cov=./ || exit /b %errorlevel%
	python setup.py sdist bdist_wheel || exit /b %errorlevel%
	python -m twine check dist/* || exit /b %errorlevel%
) else if "%1" == submitcoverage (
	venv\Scripts\activate || exit /b %errorlevel%
	python -m codecov || exit /b %errorlevel%
) else (
	echo Invalid option; must be either clean, dev_install, cython_install,
	echo install, test, submitcoverage, deactivate, or activate
)
