# imports
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import math

import random
import timeit

def main():
        
        model = input("Model name: ")
        typ = input("Data type: ")
        
        directory = "../MNIST_Training/" + model + "_" + typ + "/mc_"+ typ + "_" + model + "_"


        accuracy = np.zeros((11, 100))
        for i in range(11):

                accuracy[i] = np.load(directory + str(i*10) + ".npz", allow_pickle=True)['acc']

        indices = np.arange(0, 101, 10)
        
        sample_size = np.shape(accuracy)[1]
        avg_acc = np.sum(accuracy, axis=1)
        avg_acc = np.divide(avg_acc, sample_size)
                
        error = np.std(accuracy, axis = 1)

        if typ == "dead":
                message = "Inputs Removed from Network"
                ax = "deleted"

        else:
                message = "Inputs Perturbed"
                ax = "perturbed"

        #plt.ylim(0.8, 0.9)
        plt.errorbar(indices, avg_acc, error, ecolor='r', capsize=3, fmt="k--.")
        plt.xlabel("Number of inputs " + ax)
        plt.ylabel(f"Accuracy over {sample_size} simulations")
        plt.title("Accuracy on " + model.upper() +  " Model with " + message)
        plt.show()

        """
        loader_dead = np.load(directory_dead + "_all.npz", allow_pickle = True)
        acc_dead = loader_dead['acc'] # num_deleted x num_samples
        indices = loader_dead['ind']

        loader_noisy = np.load(directory_noisy +  "_all.npz", allow_pickle = True)
        acc_noisy = loader_noisy['acc'] # num_perturbed x num_samples
        
        
        sample_size = np.shape(acc_dead)[1]
        
        avg_acc = np.sum(acc_dead, axis = 1)
        avg_acc = np.divide(avg_acc, sample_size)

        error = np.std(acc_dead, axis = 1)
        
        #plt.plot(np.arange(num_deleted), avg_acc)
        plt.ylim(0.8, 0.9)
        plt.errorbar(indices, avg_acc, error, ecolor='r', capsize=3, fmt="k--.")
        plt.xlabel("Number of inputs deleted")
        plt.ylabel(f"Accuracy over {sample_size} simulations")
        plt.title("MNIST Classification Accuracy on " + model.upper() +  " Model with Inputs Removed from Network")
        plt.show()

        
        #plt.ylim(0.8, 0.9)
        plt.errorbar(indices, avg_acc, error, ecolor='r', capsize=3, fmt="k--.")
        plt.xlabel("Number of inputs deleted")
        plt.ylabel(f"Accuracy over {sample_size} simulations")
        plt.title("MNIST Classification Accuracy on " + model.upper() +  " Model with Inputs Removed from Network")
        plt.show()
        # ------------------------------------------------------------------

        sample_size = np.shape(acc_noisy)[1]
        num_perturbed = np.shape(acc_noisy)[0]
        avg_acc = np.sum(acc_noisy, axis = 1)
        avg_acc = np.divide(avg_acc, sample_size)

        error = np.std(acc_noisy, axis=1)

        #plt.plot(np.arange(num_deleted), avg_acc)
        #plt.ylim(0.8, 0.9)
        plt.errorbar(indices, avg_acc, error, ecolor='r', capsize=3, fmt="k--.")
        plt.xlabel("Number of inputs perturbed")
        plt.ylabel(f"Accuracy over {sample_size} simulations")
        plt.title("MNIST Classification Accuracy on " + model.upper() +  " Model with Inputs Perturbed")
        plt.ylim(0, 1)
        plt.show()
        """


if __name__ == '__main__':
        main()
