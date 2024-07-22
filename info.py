def info(train_data, test_data):
  print(f'Training set length: {len(train_data)}')
  print(f'Test set length: {len(test_data)}')
  print(f'Labels: {train_data.classes}')
  return