import numpy as np
import matplotlib.pyplot as plt
a = np.array([0, 0, 1, 1])
b = np.array([0, 1, 0, 1])

# y_and = np.array([[0, 0, 0, 1]])
y_xor = np.array([[0,1,1,0]])


total_input = []

total_input = [a, b]

total_input = np.array(total_input)

# Define the sigmoid activation function:
def sigmoid (x):
    return 1/(1 + np.exp(-x))

# Define the number of neurons
input_neurons, hidden_neurons, output_neurons = 2, 2, 1

# Total training examples
samples = total_input.shape[1]

# Learning rate
lr = 0.1

# Define random seed to replicate the outputs
np.random.seed(42)

# Initializing the weights for hidden and output layers

w1 = np.random.rand(hidden_neurons, input_neurons)
w2 = np.random.rand(output_neurons, hidden_neurons)

# Forward propagation
def forward_prop(w1, w2, x):
    z1 = np.dot(w1, x)
    a1 = sigmoid(z1)
    z2 = np.dot(w2, a1)
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

# Backward propagation
def back_prop(m, w1, w2, z1, a1, z2, a2, y):

    dz2 = (a2 - y) * a2 * (1 - a2)
    dw2 = np.dot(dz2, a1.T)
    dz1 = np.dot(w2.T, dz2) * a1*(1-a1)
    dw1 = np.dot(dz1, total_input.T)
   
    dw1 = np.reshape(dw1, w1.shape)
    dw2 = np.reshape(dw2,w2.shape)
    return dz2,dw2,dz1,dw1

losses = []
epochs = 100000

for i in range(epochs):
    z1, a1, z2, a2 = forward_prop(w1, w2, total_input)
    loss = (1/(2*samples)) * np.sum((a2 - y_xor) ** 2)
    losses.append(loss)
    da2, dw2, dz1, dw1 = back_prop(samples, w1, w2, z1, a1, z2, a2, y_xor)
    w2 = w2-lr*dw2
    w1 = w1-lr*dw1

# We plot losses to see how our network is doing
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()

# Creating the predict function

def predict(w1,w2,input):
    z1, a1, z2, a2 = forward_prop(w1,w2,test)
    a2 = np.squeeze(a2)

    if a2>=0.5:
        print("For input", [i[0] for i in input], "output is 1")
    else:
        print("For input", [i[0] for i in input], "output is 0")


test = np.array([[0],[0]])
predict(w1,w2,test)

test = np.array([[0],[1]])
predict(w1,w2,test)

test = np.array([[1],[0]])
predict(w1,w2,test)

test = np.array([[1],[1]])
predict(w1,w2,test)