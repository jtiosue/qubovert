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
    python -m pip install --user virtualenv || goto error
	python -m virtualenv venv || goto error
	venv\Scripts\activate || goto error
	python -m pip install --upgrade pip || goto error
	pip install -e . || goto error
	pip install -r requirements-dev.txt || goto error
) else if "%1" == "install" (
    python -m pip install --user virtualenv || goto error
	python -m virtualenv venv || goto error
	venv\Scripts\activate || goto error
	python -m pip install --upgrade pip || goto error
	pip install -e .
) else if "%1" == "cython_install" (
	rem use this if you want to use cython to recrete the c file from the pyx file
	python -m pip install --user virtualenv || goto error
	python -m virtualenv venv || goto error
	venv\Scripts\activate || goto error
	python -m pip install --upgrade pip || goto error
	pip install -r requirements-dev.txt || goto error
	pip install -e .
) else if "%1" == "test" (
	venv\Scripts\activate || goto error
	python -m pydocstyle convention=numpy qubovert || goto error
	python -m pytest --codestyle --cov=./ || goto error
	python setup.py sdist bdist_wheel || goto error
	python -m twine check dist/* || goto error
) else if "%1" == submitcoverage (
	venv\Scripts\activate || goto error
	python -m codecov || goto error
) else (
	echo Invalid option; must be either clean, dev_install, cython_install,
	echo install, test, submitcoverage, deactivate, or activate
)

goto end

:error
exit /b %errorlevel%

:end
