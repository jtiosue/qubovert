function pause() {
    read -p "PAUSED"
}

python -m pydocstyle convention=numpy qubovert
pause
python -m pytest --codestyle
pause
python setup.py sdist bdist_wheel
pause
python -m twine check dist/*
pause
