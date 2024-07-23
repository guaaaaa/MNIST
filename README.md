A simple 3-layer CNN model for the MNIST training set

Training set accurcy: 97.01%
Test set accuracy: 97.76%

# To play with the model
## 1. Clone the repo to a local directory
## 2. Create a virtual environment and activate it
```make venv```\\
```source mnist/bin/activate```

## 3. Install dependencies
```make install```

## 4. To train the model
```make train```

## 5. To evaluate the model
```make eval```

# Customize training
```python train.py --epochs 3 --batch_size 32 --lr 0.01```
You can change the epochs, batch size, and learning rate to desired values

## If you want to save a trained model
```python train.py --epochs 3 --batch_size 32 --lr 0.01 --save True```
Note: only one customized model will be saved. If a customized model already exist, this command will override the previous model

## If you want to evaluate a customized model
```python eval.py --load True```
