python -m pydocstyle convention=numpy qubovert
python -m pytest --codestyle --cov=./
python setup.py sdist bdist_wheel
python -m twine check dist/*
rmdir /S /Q dist
rmdir /S /Q build
rmdir /S /Q qubovert.egg-info
