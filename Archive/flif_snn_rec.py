# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import plotsrc
import Fractional_LIF_rec
import math
import random

import timeit


class SNN(nn.Module):
        def __init__(self, num_input, num_hidden, num_output, num_steps, device, dt=0.1, injection=10):
                super().__init__()

                # positive forward connections input -> hidden
                self.forward_hidden = nn.Linear(num_input, num_hidden)
                self.forward_hidden.weight = nn.Parameter(torch.abs(self.forward_hidden.weight))

                # positive forward connections hidden -> output
                self.forward_output = nn.Linear(num_hidden, num_output)
                self.forward_output.weight = nn.Parameter(torch.abs(self.forward_output.weight))

                # feedback connections output -> hidden
                self.feedback = nn.Linear(num_output, num_hidden)
                self.feedback.weight = nn.Parameter(torch.zeros_like(self.feedback.weight))

                # negative lateral connections hidden -> hidden; hidden neurons do not contribute to themselves (0 diagonal)
                self.lateral_hidden = nn.Linear(num_hidden, num_hidden)
                weight = torch.ones_like(self.lateral_hidden.weight) * -1
                weight.fill_diagonal_(0)
                self.lateral_hidden.weight = nn.Parameter(weight)

                # negative lateral connections output -> output; output neurons do not contribute to themselves (0 diagonal)
                self.lateral_output = nn.Linear(num_output, num_output)
                weight = torch.ones_like(self.lateral_output.weight) * -1
                weight.fill_diagonal_(0)
                self.lateral_output.weight = nn.Parameter(weight)
                

                self.hidden_layer = Fractional_LIF_rec.FLIF(num_hidden, device, num_steps)
                self.output_layer = Fractional_LIF_rec.FLIF(num_output, device, num_steps)


                self.input_a = list()
                self.input_theta = list()
                self.hidden_a = list()
                self.hidden_theta = list()
                self.output_a = list()
                self.output_theta = list()
                self.no_count = True

                
                self.num_steps = num_steps
                self.device = device

                self.dt = dt
                self.injection = injection

                self.alpha = 0.1
                self.beta = 0.3
                self.gamma = 0.1
                self.eta = 0.15

        def reset_fire_rates(self):

                self.input_a = list()
                self.input_theta = list()
                self.hidden_a = list()
                self.hidden_theta = list()
                self.output_a = list()
                self.output_theta = list()
                self.no_count = True
                
        def forward(self, data, targets, plot):

                tau_a = 15#self.num_steps/5 #(40)
                tau_theta = 150#self.num_steps*2 (400)

                batch = data.size(0)

                output_mem_trace = list()
                output_spikes_trace = list()
                hidden_spikes_trace = list()
                hidden_mem_trace = list()

                input_vector = spikegen.to_one_hot(targets, 10) * self.injection

                if not self.training:
                        input_vector = input_vector * 0


                hidden_mem = self.hidden_layer.init_mem(batch)
                output_mem = self.output_layer.init_mem(batch)

                hidden_spikes = torch.zeros_like(hidden_mem)
                output_spikes = torch.zeros_like(output_mem)

                self.reset_fire_rates()
                """
                torch.cuda.empty_cache()
                t = torch.cuda.get_device_properties(0).total_memory
                r = torch.cuda.memory_reserved(0)
                a = torch.cuda.memory_allocated(0)
                print(f"For Item {item}: Free mem: {(r-a)/1000000}MB. Allocated: {a/1000000}MB. Mem remaining: {(t-a)/1000000}MB")
                """
                        
                for step in range(self.num_steps):

                                
                        input_spikes = data[:, step, :]
                        
                        # compute current to hidden layer:
                        # forward from input
                        # feedback from output
                        # lateral connections
                        hidden_current = self.forward_hidden(input_spikes)
                        hidden_current += self.feedback(output_spikes)
                        hidden_current += self.lateral_hidden(hidden_spikes)

                        # Stimulate hidden layer:
                        hidden_spikes, hidden_mem = self.hidden_layer(hidden_current, hidden_mem)
                        hidden_spikes_trace.append(hidden_spikes)
                        hidden_mem_trace.append(hidden_mem)

                        # compute current to output layer:
                        # forward from hidden
                        # lateral connections
                        # supervised current from targets (if training = True)
                        output_current = self.forward_output(hidden_spikes)
                        output_current += self.lateral_output(output_spikes)

                        output_current += input_vector

                        # Stimulate output layer:
                        output_spikes, output_mem = self.output_layer(output_current, output_mem)

                        output_mem_trace.append(output_mem)
                        output_spikes_trace.append(output_spikes)

                        # Update a's and thetas
                        if self.no_count:
                                self.input_a.append(input_spikes)
                                self.input_theta.append(input_spikes)
                                        
                                self.hidden_a.append(hidden_spikes)
                                self.hidden_theta.append(hidden_spikes)

                                self.output_a.append(output_spikes)
                                self.output_theta.append(output_spikes)

                                self.no_count = False

                        else:
                                
                                self.input_a.append((1 - self.dt/tau_a) * self.input_a[step-1] + input_spikes / tau_a)
                                self.input_theta.append((1 - self.dt/tau_theta) * self.input_theta[step-1] + input_spikes / tau_theta)
                                        
                                self.hidden_a.append((1 - self.dt/tau_a) * self.hidden_a[step-1] + hidden_spikes / tau_a)
                                self.hidden_theta.append((1 - self.dt/tau_theta) * self.hidden_theta[step-1] + hidden_spikes / tau_theta)

                                self.output_a.append((1 - self.dt/tau_a) * self.output_a[step-1] + output_spikes / tau_a)
                                self.output_theta.append((1 - self.dt/tau_theta) * self.output_theta[step-1] + output_spikes / tau_theta)
                        

                # steps x batch x layer
                self.input_a = torch.stack(self.input_a)
                self.input_theta = torch.stack(self.input_theta)
                self.hidden_a = torch.stack(self.hidden_a)
                self.hidden_theta = torch.stack(self.hidden_theta)
                self.output_a = torch.stack(self.output_a)
                self.output_theta = torch.stack(self.output_theta)


                # record spikes for weights
                self.hidden_spikes = torch.stack(hidden_spikes_trace)
                self.output_spikes = torch.stack(output_spikes_trace)

                output_mem_trace = torch.stack(output_mem_trace)

                """
                sample = random.randint(0, 127)
                fig, ax = plt.subplots(2)
                ax[0].pl
                """
                if plot:

                        
                        file_location = "MNIST_Training/post_train_rec2.npz"

                        hid_spk = self.hidden_spikes.detach().cpu().numpy()
                        out_spk = self.output_spikes.detach().cpu().numpy()
                        out_mem = output_mem_trace.detach().cpu().numpy()
                        hid_mem = torch.stack(hidden_mem_trace).detach().cpu().numpy()
                        
                        np.savez(file_location, hid_spk=hid_spk, hid_mem=hid_mem, out_spk = out_spk, out_mem=out_mem)
                
                return self.output_spikes, output_mem_trace

        # helper function for modified matrix multiplication. The result is
        # a product of the products of elements + c of mat1 and mat2
        # Needed for computing updates to feedback weights
        def mat_mod_mul(self, mat1, mat2, c):

                dim0 = mat1.size(0)
                dim1 = mat2.size(1)

                result = torch.zeros(dim0, dim1)

                for i in range(dim1):
                        """
                        for j in range(dim1):

                                row = mat1[i]
                                col = mat2[:, j]

                                intermed = torch.mul(row, col)
                                intermed = intermed + c

                                result[i, j] = torch.prod(intermed)
                        """
                        row = torch.mul(mat1, mat2[:, i]) + c
                        row = torch.prod(row, dim=1)
                        result[:,i] = row

                return result
        
        def weight_update(self):
                
                tau = 10 # NOTE: seconds, may need to be 10000ms instead
                # spikes is (steps x batch x output_layer)
                # a's and thetas are (steps x batch x layer)
                
                # Using spikes from forward, update weights
                # forward weights
                input_delta = self.input_a - self.input_theta # (a_i - theta_i)
                hidden_delta = self.hidden_a - self.hidden_theta # (a_j - theta_j)
                output_delta = self.output_a - self.output_theta # (a_k - theta_k)


                
                c = 1#(1 - self.dt/tau)
                exps = np.arange(self.num_steps)
                coeffs = np.power(c, exps)
                
                hidden_size = self.forward_hidden.weight.size(0)
                output_size = self.forward_output.weight.size(0)

                
                coeffs_hidden = torch.transpose(torch.broadcast_to(torch.tensor(coeffs), (hidden_size, self.num_steps)), 0, 1).float().to(self.device) # coefficient matrix with hidden layer size
                coeffs_output = torch.transpose(torch.broadcast_to(torch.tensor(coeffs), (output_size, self.num_steps)), 0, 1).float().to(self.device) # coefficient matrix with output layer size

                batch = self.output_spikes.size(1)
                
                for image in range(batch):

                        # forward weights:

                        # feedforward weights from input to hidden layer
                        weight = self.forward_hidden.weight
                        new_weight = torch.mul(self.hidden_spikes[:, image, :], coeffs_hidden)
                        new_weight = torch.matmul(torch.transpose(new_weight, 0, 1), input_delta[:, image, :])
                        new_weight = new_weight * self.alpha * self.dt
                        new_weight = new_weight + (weight * c**self.num_steps)
                        self.forward_hidden.weight = nn.Parameter(new_weight)


                        # feedforward weights from hidden to output layer
                        weight = self.forward_output.weight
                        new_weight = torch.mul(self.output_spikes[:, image, :], coeffs_output)
                        new_weight = torch.matmul(torch.transpose(new_weight, 0, 1), hidden_delta[:, image, :])
                        new_weight = new_weight * self.alpha * self.dt
                        new_weight = new_weight + (weight * c**self.num_steps)
                        self.forward_output.weight = nn.Parameter(new_weight)


                        # lateral output weights
                        weight = self.lateral_output.weight
                        new_weight = torch.mul(self.output_spikes[:, image, :], coeffs_output)
                        new_weight = torch.matmul(torch.transpose(new_weight, 0, 1), output_delta[:, image, :])
                        new_weight = new_weight * self.gamma * -1 * self.dt
                        new_weight = new_weight + (weight * c**self.num_steps)
                        new_weight.fill_diagonal_(0)
                        self.lateral_output.weight = nn.Parameter(new_weight)
                        

                        # lateral hidden weights
                        weight = self.lateral_hidden.weight
                        new_weight = torch.mul(self.hidden_spikes[:, image, :], coeffs_hidden)
                        new_weight = torch.matmul(torch.transpose(new_weight, 0, 1), hidden_delta[:, image, :])
                        new_weight = new_weight * self.eta * self.dt
                        new_weight = new_weight + (weight * c**self.num_steps)
                        new_weight.fill_diagonal_(0)
                        self.lateral_hidden.weight = nn.Parameter(new_weight)


                        # feedback weights
                        weight = self.feedback.weight
                        mat1 = torch.transpose(torch.mul(hidden_delta[:, image, :], self.beta*self.dt), 0, 1)
                        mat2 = self.output_spikes[:, image, :]
                        new_weight = self.mat_mod_mul(mat1, mat2, c).to(self.device)
                        new_weight = torch.mul(new_weight, weight)
                        self.feedback.weight = nn.Parameter(new_weight)

                
