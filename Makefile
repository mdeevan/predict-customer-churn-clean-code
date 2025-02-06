install:
	python -m pip install -r requirements_py3.10.txt

test:
	pytest churn_script_logging_and_tests_class.py

test_module:
	pytest churn_script_logging_and_tests.py

run:
	python churn_library_class.py

run_module:
	python churn_library.py

format: 
	autopep8 --in-place --aggressive churn_library_class.py
	autopep8 --in-place --aggressive churn_script_logging_and_tests_class.py

format_module:
	autopep8 --in-place --aggressive churn_library.py
	autopep8 --in-place --aggressive churn_script_logging_and_tests.py

format_all: format format_module

lint: format
	pylint churn_library_class.py

lint_module:
	pylint churn_library.py

lint_test: format
	pylint churn_library_class.py

lint_test_module:
	pylint churn_library.py

lint_all: lint, lint_module, lint_test, lint_test_module

