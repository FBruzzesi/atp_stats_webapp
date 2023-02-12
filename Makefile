init-env:
	pip install . --no-cache-dir

clean-notebooks:
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb

clean-folders:
	rm -rf .ipynb_checkpoints __pycache__ .pytest_cache */.ipynb_checkpoints */__pycache__ */.pytest_cache
	rm -rf site build dist htmlcov .coverage

interrogate:
	interrogate -vv --ignore-nested-functions --ignore-module --ignore-init-method --ignore-private --ignore-magic --ignore-property-decorators --fail-under=90 webapp atp_stats

style:
	isort --profile black -l 90 webapp atp_stats
	black --target-version py38 --line-length 90 webapp atp_stats

test:
	pytest tests -vv

test-coverage:
	coverage run -m pytest
	coverage report -m

check: interrogate style clean-folders

docker-build:
	docker build -t atp-webapp -f Dockerfile .

	docker container prune --force
	docker image prune --force

	docker images | grep atp-webapp
