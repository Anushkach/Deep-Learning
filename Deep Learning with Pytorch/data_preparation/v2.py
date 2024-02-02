
torch.manual_seed(42)

# Build tensors from np arrays
X_tensor=torch.as_tensor(x).float()
y_tensor=torch.as_tensor(y).float()

# build dataset containing all data
dataset=TensorDataset(X_tensor,y_tensor)

# perform split
split_ratio=0.8
n_total=len(dataset)
n_train=int(n_total*split_ratio)
n_val=n_total-n_train

# perform train-test split in pytorch
train_data,val_data=random_split(dataset,[n_train,n_val])

train_loader=DataLoader(dataset=train_data,batch_size=16,shuffle=True)
val_loader=DataLoader(dataset=val_data,batch_size=16) # we dont use shuffle for validation data because we want to keep the order
