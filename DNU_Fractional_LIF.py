# imports
import snntorch as snn

import torch
import torch.nn as nn

import numpy as np
import math


import ray


class FLIF(nn.Module):


        weight_vector = list()

        def __init__(self, size, identity, alpha=0.15, dt=0.1, threshold=-50, V_init=-70, VL=-70, V_reset=-70, gl=0.025, Cm=0.5):

                super(FLIF, self).__init__()

                self.alpha = alpha
                self.dt = dt
                self.threshold = threshold
                self.V_init = V_init
                self.VL = VL
                self.V_reset = V_reset
                self.gl = gl
                self.Cm = Cm
                self.tau = Cm/gl
                
                self.delta_trace = dict()

                for i in range(size):
                        self.delta_trace.update({i:list()})
                
                self.V_trace = [0]*size
                self.N = 0
                
                self.spike_gradient = ATan.apply
                self.layer_size = size
                
                # precompute x weight values
                if len(FLIF.weight_vector) == 0:
                        x = 5000
                        nv = np.arange(x-1)
                        FLIF.weight_vector = (x-nv+1)**(1-self.alpha)-(x-nv)**(1-self.alpha)
                        


        # DEPRECATED: I and V_old are num_steps x layer size tensors
        # I_old is a (num_steps*batch_size) x layer_size tensor
        # Need to modify the values in mem and return it
        def forward(self, I_old):

                
                size = self.layer_size
                I = I_old.tolist() # given i, the ith chunk of this matrix is I[:,start:end]

                input_size = I_old.size()[1]

                
                # Compute additional weights if necessary
                if self.N + input_size >= len(FLIF.weight_vector):
                        print("Generating additional weights...")
                        new_vals = list()
                        for i in range(1, 1000):
                                index = self.N + i
                                weight = (index + 1)**(1-self.alpha) - index**(1-self.alpha)
                                new_vals.append(weight)

                        FLIF.weight_vector = np.concatenate((new_vals, FLIF.weight_vector), axis = 0)


                # Computations for batch
                ray.init()
                parameters = ray.put([self.threshold, self.V_reset, self.VL, self.dt, self.tau, self.gl, self.alpha, self.Cm])
                old_N = ray.put(self.N)
                weights = ray.put(FLIF.weight_vector)
                V_trace = ray.put(self.V_trace)
                delta = ray.put(self.delta_trace)

                # output is a list of tuples; each tuple is an index, list of spikes over the duration, and list of membrane potentials
                output = ray.get([FLIF.single_neuron.remote(I[i], i, parameters, old_N, V_trace, delta, weights) for i in range(size)])

                spike_matrix = [[0]*input_size]*size
                trace_matrix = [[0]*input_size]*size
                
                
                self.N += input_size # added this many steps
                for i in range(size):
                
                        index = output[i][0]
                        spikes = output[i][1]
                        new_trace = output[i][2]
                        
                        # Update my delta_trace matrix
                        new_delta = np.subtract(new_trace[1:], new_trace[0:input_size-1])
                        old_delta = self.delta_trace.get(index)
                        if len(old_delta) > 0:
                                new_delta = np.concatenate(old_delta, new_delta)
                                
                        self.delta_trace.update({index : new_delta})

                        spike_matrix[index] = spikes
                        trace_matrix[index] = new_trace

                
                
                trace_tensor = torch.tensor(trace_matrix)
                spike_tensor = torch.tensor(spike_matrix)


                self.spike_gradient(trace_tensor - self.threshold)

                # format and return spikes
                ray.shutdown()

                #if zip_data:

                        #return FLIF.zipper(spike_tensor), FLIF.zipper(trace_tensor)
                
                
                return spike_tensor, trace_tensor


        # parallelized version calculates spikes and trace for a single neuron
        # Given a single neuron and a sequence of input currents (batch_size*num_steps), calculate the trace history
        # I_sequence is a (batch_size*num_steps) x 1 list
        # Returns (batch_size*num_steps) x 1 lists of traces and spikes
        @ray.remote
        def single_neuron(I_sequence, index, parameters, old_N, V_trace, delta_trace_dict, weights):

                num_output = len(I_sequence)
                
                spikes = [0]*num_output
                trace = [0]*num_output
                # use self.delta_trace.get(index) as a func parameter
                #delta_trace = delta.get(index) 
                
                V = torch.tensor(V_trace[index])

                delta_trace = delta_trace_dict.get(index)

                N = old_N

                # parameters: threshold, V_reset, dt, tau, gl, alpha, Cm
                threshold = parameters[0]
                V_reset = parameters[1]
                VL = parameters[2]
                dt = parameters[3]
                tau = parameters[4]
                gl = parameters[5]
                alpha = parameters[6]
                Cm = parameters[7]

                
                for I in I_sequence:
                        match N:
                                case 0:
                                        V_new = torch.ones(1)*-70
                                        spike = ((V - threshold) > 0).float()
                                        reset = (spike * (threshold - V_reset)).detach()

                                        V_new = torch.sub(V_new, reset)
                                        trace[N] = V_new.item()
                                        spikes[N] = spike.item()

                                        V = V_new
                                        N += 1
                                        continue

                                case 1:
                                
                                       V_new = V + (dt/tau)*(-1 * V + I/gl)

                                case _:
 
                                       V_new = dt**(alpha) * math.gamma(2-alpha) * (-gl*(V-VL)+I) / Cm + V

                                       memory_V = np.inner(weights[-N+1:], delta_trace)
                                       memory_tensor = torch.tensor(memory_V)

                                       V_new = torch.sub(V_new, memory_tensor)

                        spike = ((V - threshold) > 0).float()
                        reset = (spike * (threshold - V_reset)).detach()

                        V_new = torch.sub(V_new, reset)
                        delta  = torch.sub(V_new, V)
                        delta_trace.append(delta.item())
                        
                        V = V_new
                        trace[N] = V_new.item()
                        spikes[N] = spike.item()
                        N += 1

                N -= 1
                        
               
                return (index, spikes, trace)
        
        
        def reset(self):
                for i in range(self.layer_size):
                        self.delta_trace.update({i:list()})
                        
                self.N = 0

        def init_mem(self):

                return torch.zeros(0)

        # Fix detachment issues;
        # This function takes in a tensor 'mem' and a history of values
        # Basically just push all the history values into mem and call
        # spike_gradient on them
        def iterate_grad(self, mem, history, num_steps, batch_size):
        
                # history has dimensions batch_size*num_steps x layer_size 
                # need to break along second dimension; each slice will be set to mem
                # then passed to self.spike_gradient
                for i in range(num_steps*batch_size):
                        #mem = 0 + torch.narrow(history, 0, i*batch_size, batch_size)

                        if i == 0:
                                mem = torch.ones(self.layer_size, requires_grad=True)*-70
                        else:
                                delta = [self.delta_trace.get(j)[i-1] for j in range(self.layer_size)]
                                mem = mem + torch.tensor(delta)

                        self.spike_gradient(mem - self.threshold)

        # util function to 'zip' data back into a 3 dimensional tensor
        def zipper(data, num_steps):

                zipped_data = list()

                length = int(data.size()[1] / num_steps)
                
                for i in range(num_steps):

                        data_slice = torch.narrow(data, 1, i*length, length)

                        zipped_data.append(data_slice)

                return torch.stack(zipped_data)
                
class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
                spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
                ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
                return spk

        @staticmethod
        def backward(ctx, grad_output):
                (mem, ) = ctx.saved_tensors  # retrieve the membrane potential
                print(mem.size())
                grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
                return grad
