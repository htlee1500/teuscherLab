# imports
import snntorch as snn

import torch
import torch.nn as nn

import numpy as np
import math


import ray
import timeit

import os

class FLIFP(nn.Module):


        weight_vector = list()

        def __init__(self, size, alpha=0.2, dt=0.1, threshold=-50, V_init=-70, VL=-70, V_reset=-70, gl=0.025, Cm=0.5):

                super(FLIFP, self).__init__()

                self.alpha = alpha
                self.dt = dt
                self.threshold = threshold
                self.V_init = V_init
                self.VL = VL
                self.V_reset = V_reset
                self.gl = gl
                self.Cm = Cm
                self.tau = Cm/gl
                
                
                self.spike_gradient = ATan.apply
                self.layer_size = size
                
                # precompute x weight values
                if len(FLIFP.weight_vector) == 0:
                        x = 2000
                        nv = np.arange(x-1)
                        FLIFP.weight_vector = (x-nv+1)**(1-self.alpha)-(x-nv)**(1-self.alpha)
                        


        # DEPRECATED: I and V_old are num_steps x layer size tensors
        # Take imported currents tensor and compute changes to each neuron in parallel
        def forward(self, I):

                cores = os.cpu_count()
                # break layer size into [cores] groups, for parallelization
                group_size = math.floor(self.layer_size / cores)

                indices = list()
                for i in range(cores-1):

                        indices.append((group_size * i, group_size * (i + 1)))

                indices.append((group_size * (cores-1), self.layer_size))
        

                # after parallelizing, stitch all the spikes and voltages back together
                # run the voltages through the gradient for backprop
                ray.init(include_dashboard = False, configure_logging = True, logging_level = "error")
                weights = ray.put(torch.tensor(FLIFP.weight_vector))
                I_all = ray.put(I)
                parameters = ray.put([self.alpha, self.dt, self.threshold, self.V_init, self.VL, self.V_reset, self.gl, self.Cm])                
                start = timeit.default_timer()

                """
                voltages = ray.get([FLIFP.single_neuron.remote(i, parameters, I_all, weights) for i in range(self.layer_size)]) # [layer_size] count of [step] x [batch size] tensors
                end = timeit.default_timer()

                print("Parallel processing in", end-start, "seconds")
                
                ray.shutdown()
                
                spikes = list()

                
                for i in range(self.layer_size):

                        spks = self.spike_gradient(voltages[i] - self.threshold)
                        spikes.append(spks)
                

                # turn these back into 3d tensors
                voltages = torch.stack(voltages, dim=0) # layer x step x batch, needs to be batch x steps x layer
                spikes = torch.stack(spikes, dim = 0)

                voltages = torch.transpose(voltages, 0, 2)
                spikes = torch.transpose(spikes, 0, 2)
                """
                voltages = ray.get([FLIFP.neuron_group.remote(indices[i], parameters, I_all, weights) for i in range(cores)]) #[cores] count of [step] x [batch size] x group size tensors
                end = timeit.default_timer()

                print("Parallel processing in", end-start, "seconds")
                
                ray.shutdown()

                # stitch groups back together
                stitch = timeit.default_timer()
                voltage_hist = voltages[0]
                for i in range(1, cores):

                        voltage_hist = torch.cat((voltage_hist, voltages[i]), dim=2)

                in_time = timeit.default_timer()

                steps = voltage_hist.size(0)
                spikes = list()
                """
                for i in range(steps):

                        mem_at_step = voltage_hist[i,:,:]
                        spike = self.spike_gradient(mem_at_step - self.threshold)
                        spikes.append(spike)
                
                
                spike_calc = timeit.default_timer()

                spikes = torch.stack(spikes) # steps x batch x layer
                
                return torch.transpose(spikes, 0, 1), torch.transpose(voltage_hist, 0, 1)
                """

                spikes = self.spike_gradient(voltage_hist - self.threshold)


                return torch.transpose(spikes, 0, 1), torch.transpose(voltage_hist, 0, 1)
                
                

        # Each neuron calculates its own spikes and membrane potential history; returns both to the forward function
        # Parameters: 0-alpha, 1-dt, 2, threshold
        @ray.remote
        def single_neuron(index, parameters, I_all, weights):

                I = I_all[:,:,index]
                steps = I_all.size(1)
                V_trace = list()
                delta_trace = list()

                # Parameters
                alpha = parameters[0]
                dt = parameters[1]
                threshold = parameters[2]
                V_init = parameters[3]
                VL = parameters[4]
                V_reset = parameters[5]
                gl = parameters[6]
                Cm = parameters[7]

                # N = 0
                V_old = torch.ones(I_all.size(0))*V_init
                V_trace.append(V_old)

                # N = 1
                tau = Cm / gl
                V_new = V_old + (dt/tau)*(-1 * V_old + I[:,0]/gl)
                delta_trace.append(V_new - V_old)
                V_old = V_new
                V_trace.append(V_new)
                

                # N >= 2
                for N in range(2, steps):

                        spikes = ((V_old - threshold) > 0).float()

                        V_new = dt**(alpha) * math.gamma(2-alpha) * (-gl*(V_old-VL)+I[:,N]) / Cm + V_old

                        delta = torch.stack(delta_trace) # steps x batch
                        delta = torch.transpose(delta, 0, 1) # batch x steps

                        memory_V = torch.matmul(delta, weights[-N+1:].float())

                        V_new = torch.sub(V_new, memory_V)
                        delta_trace.append(V_new - V_old)

                        reset = (spikes * (threshold - V_reset)).detach()
                        V_new = torch.sub(V_new, reset)
                        V_trace.append(V_new)
                        V_old = V_new

                return torch.stack(V_trace)

        @ray.remote
        def neuron_group(indices, parameters, I_all, weights):

                start_index = indices[0]
                end_index = indices[1]
                
                I = I_all[:,:,start_index:end_index]
                steps = I_all.size(1)
                V_trace = list()
                delta_trace = list()

                # Parameters
                alpha = parameters[0]
                dt = parameters[1]
                threshold = parameters[2]
                V_init = parameters[3]
                VL = parameters[4]
                V_reset = parameters[5]
                gl = parameters[6]
                Cm = parameters[7]

                # N = 0
                V_old = torch.ones_like(I[:,0,:])*V_init
                V_trace.append(V_old)

                # N = 1
                tau = Cm / gl
                V_new = V_old + (dt/tau)*(-1 * V_old + I[:,0,:]/gl)
                delta_trace.append(torch.sub(V_new, V_old).detach())
                V_old = V_new
                V_trace.append(V_new)
                

                # N >= 2
                for N in range(2, steps):

                        spikes = ((V_old - threshold) > 0).float()

                        V_new = dt**(alpha) * math.gamma(2-alpha) * (-gl*(V_old-VL)+I[:,N,:]) / Cm + V_old

                        delta = torch.stack(delta_trace) # steps x batch x group
                        delta = torch.transpose(delta, 2, 0) # group x batch x steps
                        delta = torch.transpose(delta, 0, 1) # batch x group x steps

                        memory_V = torch.matmul(delta, weights[-N+1:].float())

                        V_new = torch.sub(V_new, memory_V)
                        delta_trace.append(torch.sub(V_new, V_old).detach())

                        reset = (spikes * (threshold - V_reset)).detach()
                        V_new = torch.sub(V_new, reset)
                        V_trace.append(V_new)
                        V_old = V_new

                return torch.stack(V_trace)
                        
                        

                
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
