# Computer-Vision
a two-layer nerual network for processing MNIST

"parasearch.py", search the best parameter: learning rate, hidden size and regularization strength.

"train.py" define simpleNetwork class including active function, bp gradient, loss, SGD, and save model. Train the model and visualize the loss can accuracy curves.


"test.py", import the model and test the model performance, visualize the parameters(W1, W2) of each layer of the neural network.

Steps:
1. Run "parasearch.py" to search parameter.
2. Uncomment the last part of "train.py" and run to train the model. 
3. Run "test.py"

Note: Some of the codes commented are different methods of the same procedure, you can switch the comments to adopt different methods.
