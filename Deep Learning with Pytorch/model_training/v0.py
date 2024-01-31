
# define numbe of epochs
n_epochs=10000

for epoch in tqdm(range(n_epochs)):

    # set model to train mode
    model.train()

    # forward pass
    y_hat=model(X_train_tensor)

    # compute loss
    loss=loss_fn(y_hat,y_train_tensor)

    # compute gradients for both b and w
    loss.backward()

    # update parameters
    optimizer.step()
    optimizer.zero_grad()
