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
	pylint ./chat.py
	flake8 ./chat.py
	isort .
test:
	pytest -v -s tests.py
build:
	virtualenv venv; \
	source ./venv/bin/activate; \
	pip install -r ./requirements.txt; \
	cp ./template.envrc .envrc
install:
	source ./venv/bin/activate; \
	pip install -r ./requirements.txt
up:
	source ./venv/bin/activate; python chat.py
linttest: lint test
