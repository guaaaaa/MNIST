import dataset, engine, info, normalize, models, loss_optimizer, utils
import argparse

# Create a parser
parser = argparse.ArgumentParser(description="Hyperparameters")

# Add arguments to parser
parser.add_argument("--epochs",
                    default=3,
                    type=int,
                    help="Number of epochs"
                    )
parser.add_argument("--batch_size",
                    default=32,
                    type=int,
                    help="Batch size"
                    )
parser.add_argument("--lr",
                    default=0.01,
                    type=float,
                    help="Learning rate"
                    )

parser.add_argument("--save",
                    default=False,
                    type=bool
                    )
# Get arguments from the parser
args = parser.parse_args()

# Define hyperparameters
epochs = args.epochs
bs= args.batch_size
lr = args.lr
save = args.save

train_data, test_data = dataset.dataset()
if train_data is None or test_data is None:
    raise Exception('Error downloading the dataset')
    
info.info(train_data, test_data)

train_data_normal, test_data_normal = normalize.normalize(train_data, test_data, bs)

device = utils.device()

model = models.MNIST_CNN().to(device)

loss_fn, optimizer = loss_optimizer.loss_optimizer(model, lr)
print("Start training")
params = engine.train(model=model, train_dataloader=train_data_normal, test_dataloader=test_data_normal, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs, device=device)

if save:
    utils.save(params)