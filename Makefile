.PHONY: init io env requirements clean check

PROJECT_DIR := $(shell pwd)
ENV = env

init: io env

check:
	@echo $(shell pwd)
io:
	mkdir io
	mkdir io/data
	mkdir io/results
	mkdir io/plots

env:
ifndef VIRTUAL_ENV
	mkdir $(ENV)
	python3 -m venv env
else
	@echo "You are already in a virtual environment."
endif

requirements:
ifdef VIRTUAL_ENV
	pip3 install -r requirements.txt
else
	@echo "Virtual environment not found. Please create it first using 'make env' followed by 'source env/bin/activate'."
endif

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*__pycache__" -delete

