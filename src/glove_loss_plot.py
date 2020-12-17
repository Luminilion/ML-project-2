import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "../data/"

loss = np.load(DATA_PATH + "loss_22epoch_250dim.npy")

epoch = [i for i in range(len(loss))]

plt.plot(epoch,loss)
plt.title("Loss Function through epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig('../plots/Loss_function_{}_epochs'.format(len(loss)))
plt.show()