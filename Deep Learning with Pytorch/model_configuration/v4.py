lr=0.1

torch.manual_seed(42)
# Create a model
model=nn.Sequential(nn.Linear(1,1))

# define SGD Optimizer
optimizer=optim.SGD(model.parameters(),lr=lr)

# Define MSE Loss function
loss_fn=nn.MSELoss(reduction='mean')
