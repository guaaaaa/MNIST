import torch
from tqdm.auto import tqdm
torch.manual_seed(42)

def accuracy_fn(y_true, y_pred):
  correct = torch.eq(y_true, y_pred).sum().item()
  acc = (correct / len(y_pred)) * 100
  return acc

def train(model, train_dataloader, test_dataloader, loss_fn, optimizer, epochs, device):
  for epoch in tqdm(range(epochs)):
    total_train_loss = 0
    total_test_loss = 0
    total_train_acc = 0
    total_test_acc = 0

    model.train()
    for X, y in train_dataloader:
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      loss = loss_fn(y_pred, y)
      total_train_loss += loss.item()
      total_train_acc += accuracy_fn(y, y_pred.argmax(dim=1))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    model.eval()
    with torch.inference_mode():
      for X, y in test_dataloader:

        X, y = X.to(device), y.to(device)
        test_pred = model(X)
        total_test_loss += loss_fn(test_pred, y)
        total_test_acc += accuracy_fn(y, test_pred.argmax(dim=1))

    print(f'Epoch: {epoch+1} | Train loss: {total_train_loss/len(train_dataloader):.5f} | Test loss: {total_test_loss/len(test_dataloader):.5f} | Train acc: {total_train_acc/len(train_dataloader):.2f}% | Test acc: {total_test_acc/len(test_dataloader):.2f}%')
  
  params = model.state_dict()
  return params