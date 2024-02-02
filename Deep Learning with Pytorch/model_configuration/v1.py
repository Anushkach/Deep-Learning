
device='cuda' if torch.cuda.is_available() else 'cpu'
lr=0.1

torch.manual_seed(42)
# create a model and send it to device
model=nn.Sequential(nn.Linear(1,1)).to(device)

# define SGD optmizer
optimizer=optim.SGD(model.parameters(),lr=lr)

# define loss function
loss_fn=nn.MSELoss(reduction='mean')

# create train step function
train_step_fn=make_train_step(model,loss_fn,optimizer)
