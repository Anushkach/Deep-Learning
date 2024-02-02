
# Define number of epochs
n_epochs=1000

losses=[]

for epoch in tqdm(range(n_epochs)):
    # inner loop
    mini_batch_losses=[]
    for x_batch,y_batch in train_loader:
        # the dataset lives on CPU, we need to send mini-batches to the device where our model lives
        x_batch=x_batch.to(device)
        y_batch=y_batch.to(device)
        
        # perform training step and return corresponding loss
        mini_batch_loss=train_step_fn(x_batch,y_batch)
        mini_batch_losses.append(mini_batch_loss)

    # compute average loss over all mini batches
    loss=np.mean(mini_batch_losses)
    losses.append(loss)
