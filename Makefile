install:
    python -m pip install -r requirements_py3.10.txt

test:
    pytest churn_script_logging_and_tests.py

run:
    python churn_library_class.py

lint:
    pylint churn_library_class.py

