import numpy as np
import matplotlib.pyplot as plt
import random
import math
import plotsrc
import torch

def main():

    #TODO: Add correctly classified identifier

    filename = input("Choose file to load: ")

    loader = np.load("../MNIST_Training/" + filename + ".npz", allow_pickle = True)
    targets = loader['tar']


    data = [0]
    while len(data) == 1:

        data_type = input("Choose data type: ")

        if data_type == 'spk':

            in_spk = loader['in_spk']
            hid_spk = loader['hid_spk']
            out_spk = loader['out_spk']
            data = [1, 2]
            
        elif data_type == 'mem':

            layer = input("Choose layer to plot: ")
            data = getData(layer, data_type, loader)

        else:
            print("Invalid data type")

    # steps x batch x layer

    if data_type == 'spk':
        # batch x steps x layer
        batch_size = np.shape(in_spk)[0]
        index = input("Choose image index: ")

        if index == 'r':

            index = random.randint(0, batch_size -1)
        
        num_steps = np.shape(in_spk)[1]

        index = int(index)

        plotsrc.plot_snn_spikes(torch.tensor(in_spk[index, :, :]), np.shape(in_spk)[2], torch.tensor(hid_spk[:, index, :]), torch.tensor(out_spk[:, index, :]), num_steps, "Spike Plot for Image with Target " + str(targets[index].item()))

        quit()
        
    
    
    num_steps = np.shape(data)[0]
    batch_size = np.shape(data)[1]
    layer_size = np.shape(data)[2]
    
    sample_size = batch_size
    dim = math.floor(math.sqrt(sample_size))

    #indices = [random.randint(0, batch_size -1) for i in range(sample_size)]
    figure, ax = plt.subplots(dim, dim, layout='constrained')


    indices = np.arange(sample_size)

    neurons = [random.randint(0, layer_size -1) for j in range(10)]
    
    for x in range(dim):

        for y in range(dim):

            if layer_size > 10:

                

                for k in range(10):
                    ax[x][y].plot(np.arange(num_steps), data[:, indices[x+y], neurons[k]], label = "Neuron " + str(neurons[k]))

                    ax[x][y].set_title("Target Digit: " + str(targets[dim*x + y]))
            
            else:
            
                for j in range(layer_size):
                    
                    ax[x][y].plot(np.arange(num_steps), data[:, indices[x+y], j], label = "Neuron " + str(j))
                    
                    ax[x][y].set_title("Target Digit: " + str(targets[dim*x + y]))
            
    plt.legend()
    plt.show()
    
        



def getData(layer, data_type, loader):

    index = layer + "_" + data_type

    try:
        data = loader[index]

    except:
        print("Invalid input type")
        return [0]

    else:
        return data

if __name__ == '__main__':
    main()
