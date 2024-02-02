
n_epochs=1000
losses=[]

for epoch in tqdm(range(n_epochs)):

    # perform train step and return corresponding loss
    loss=train_step_fn(X_train_tensor,y_train_tensor) # perform one training step
    losses.append(loss) # keep track of loss
