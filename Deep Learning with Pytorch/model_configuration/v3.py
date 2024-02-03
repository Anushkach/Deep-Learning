
device='cuda' if torch.cuda.is_available() else 'cpu'

# set learning rate
lr=0.1

torch.manual_seed(42)
# create a model and send it to device
model=nn.Sequential(nn.Linear(1,1)).to(device)

# define SGD optimizer to update parameters
optimizer=optim.SGD(model.parameters(),lr=lr)

# define mse loss function
loss_fn=nn.MSELoss(reduction='mean')

# craete train step function for our model, loss function and optimizer
train_step_fn=make_train_step(model,loss_fn,optimizer)

# create val step function for model and loss function
val_step_fn=make_val_step(model,loss_fn)

# Create summary writer to interface with tensorboard.
writer=SummaryWriter('runs/simple_linear_regression')
# fetch a single mini batch so we can use add_graph()
X_dummy,y_dummy=next(iter(train_loader))
writer.add_graph(model,X_dummy.to(device))
