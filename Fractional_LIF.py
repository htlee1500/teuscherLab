# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import math


class FLIF(nn.Module):


        weight_vector = list()

        def __init__(self, size, alpha=0.2, dt=0.1, threshold=-50, V_init=-70, VL=-70, V_reset=-70, gl=0.025, Cm=0.5):

                super(FLIF, self).__init__()

                self.alpha = alpha
                self.dt = dt
                self.threshold = threshold
                self.V_init = V_init
                self.VL = VL
                self.V_reset = V_reset
                self.gl = gl
                self.Cm = Cm
                self.V_trace = list() # trace is a list of tensors; each tensor represents voltages for all neurons
                self.spike_gradient = ATan.apply

                self.layer_size = size
                
                # precompute 2000 weight values
                if len(FLIF.weight_vector) == 0:
                        nv = np.arange(1999)
                        FLIF.weight_vector = (1999-nv)**(1-self.alpha)-(1998-nv)**(1-self.alpha)

        # Both I and V_old are (batch_size)x(layer size) tensors; need to adapt function
        def forward(self, I_old, V_old):
                I = I_old + 3
                N = len(self.V_trace)
                if N == 0:
                        V_new = torch.ones_like(I) *-70
                        self.V_trace.append(V_new)
                        return torch.zeros_like(V_new), V_new
                        
                
                spike = self.spike_gradient((V_old - self.threshold))
                reset = (spike * (self.threshold - self.V_reset)).detach()
                V_new = 0
               

                        
		# Build weight vector if needed
                w_available = len(FLIF.weight_vector)
                if N > w_available:

                        missing = N - w_available

                        for i in range(w_available, N):

                                weight = (i+2)**(1-self.alpha) - (i+1)**(1-self.alpha)
                                FLIF.weight_vector.append(weight)

                if N < 2:
                        
			#Classic LIF
                        tau = self.Cm / self.gl
                        V_new = V_old + (self.dt/tau)*(-1 * V_old + I/self.gl) # new 1x(layer size) tensor

                        self.V_trace.append(V_new)

                else:
			#Fractional LIF
                        
                        V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(V_old-self.VL)+I) / self.Cm + V_old


                        # Compute memory trace 
                        # V_trace is a list of (batch_size)x(layer_size) tensors
                        # Convert V_trace to a 3D list by first making it a tensor
                        # The result, iterable_trace, is a N x (batch_size) x (layer_size) 3D list
                        stacked_trace = torch.stack(self.V_trace, dim=0)
                        iterable_trace = stacked_trace.tolist()
                        
                        # Use iterable_trace to compute differences [V(N) - V(N-1)]
                        delta_trace = np.subtract(iterable_trace[1:], iterable_trace[0:N-1])
   
                        delta_trace = np.transpose(delta_trace, (1, 2, 0))
                        
                        # compute dot products of each neuron's delta-trace by the weight vector
                        memory_V = np.dot(delta_trace, FLIF.weight_vector[-len(self.V_trace)+1:]) # If I didn't mess up, memory_V is now a batch_size x layer_size matrix
                        memory_tensor = torch.tensor(memory_V)
                        V_new = torch.sub(V_new, memory_tensor)
                                

                V_new = torch.sub(V_new, reset)
                return spike, V_new

        def init_mem(self):
                self.V_trace.clear()
                return torch.zeros(0)
                
        

class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
                spk = (mem > 0).float() # Heaviside on the forward pass: Eq(2)
                ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
                return spk

        @staticmethod
        def backward(ctx, grad_output):
                (mem, ) = ctx.saved_tensors  # retrieve the membrane potential
                grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
                return grad
