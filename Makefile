venv:
	python -m venv ./mnist

install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

train:
	python train.py

eval:
	python eval.py

visualize:
	python visual.py