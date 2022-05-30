.PHONY: init io clean

PROJECT_DIR := $(shell pwd)
ENV = env

init: 
	mkdir io
	mkdir io/data
	mkdir io/results
	mkdir io/plots	

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "*__pycache__" -delete

