help:
	@echo 'Individual commands:'
	@echo ' lint             - Lint the code with pylint and flake8'
	@echo ' test             - Run tests'
	@echo ' build            - Setup the virtual environment and install dependencies'
	@echo ' install          - Install dependencies'
	@echo ' up               - Start ChatGPT voice chat'
	@echo ''
	@echo 'Grouped commands:'
	@echo ' linttest         - Run lint and test'
lint:
	source ./venv/bin/activate; \
	./venv/bin/pylint ./chat.py; \
	./venv/bin/flake8 ./chat.py; \
	./venv/bin/isort .
test:
	source ./venv/bin/activate; \
	./venv/bin/pytest -v -s tests.py
build:
	python3.11 -m venv venv; \
	source ./venv/bin/activate; \
	pip install -r ./requirements.txt; \
	cp ./template.envrc .envrc
install:
	source ./venv/bin/activate; \
	pip install -r ./requirements.txt
up:
	source ./venv/bin/activate; python chat.py
linttest: lint test
