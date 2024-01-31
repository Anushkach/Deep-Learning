
device= 'cuda' if torch.cuda.is_available() else 'cpu'

# set learning  rate
lr=0.1

torch.manual_seed(42)
# create model and send it to device
model=nn.Sequential(nn.Linear(1,1)).to(device)

# Define SGD optimizer to update parameters of the model
optimizer=optim.SGD(model.parameters(),lr=lr)

# define MSE loss function
loss_fn=nn.MSELoss(reduction='mean')
