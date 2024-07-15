import numpy as np
import matplotlib.pyplot as plt
import random
import math

def main():

    filename = input("Choose file to load: ")

    loader = np.load("MNIST_Training/" + filename + ".npz", allow_pickle = True)


    data = [0]
    while len(data) == 1:
        
        layer = input("Choose layer to plot: ")
        data_type = input("Choose data type: ")
        data = getData(layer, data_type, loader)

    # steps x batch x layer
    """
    batch_size = np.shape(data)[1]
    while True:
        sample_size = input("Figures to plot: ")
        try:
            sample_size = int(sample_size)

        except:
            print("Not a number.")
        else:
            if sample_size > batch_size:
                print("Too large.")
            else:
                break
    """
    
    num_steps = np.shape(data)[0]
    batch_size = np.shape(data)[1]
    layer_size = np.shape(data)[2]

    sample_size = batch_size
    dim = math.floor(math.sqrt(sample_size))

    #indices = [random.randint(0, batch_size -1) for i in range(sample_size)]
    figure, ax = plt.subplots(dim, dim)


    indices = np.arange(sample_size)
    for x in range(dim):

        for y in range(dim):

            if layer_size > 10:

                neurons = [random.randint(0, layer_size -1) for j in range(10)]

                for k in range(10):
                    ax[x][y].plot(np.arange(num_steps), data[:, indices[x+y], neurons[j]], label = "Neuron " + str(neurons[k]))

                    ax[x][y].set_title("Image No. " + str(indices[x+y]))
                    #ax[x][y].legend()
            
            else:
            
                for j in range(layer_size):
                    
                    ax[x][y].plot(np.arange(num_steps), data[:, indices[x+y], j], label = "Neuron " + str(j))
                    
                    ax[x][y].set_title("Image No. " + str(indices[x+y]))
                    #ax[x][y].legend()
            
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