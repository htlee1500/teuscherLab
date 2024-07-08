# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import math
import compute

import timeit


class FLIF(nn.Module):


        weight_vector = list()

        def __init__(self, size, alpha=0.2, dt=0.1, threshold=-65, V_init=-70, VL=-70, V_reset=-70, gl=0.025, Cm=0.5):

                super(FLIF, self).__init__()

                self.alpha = alpha
                self.dt = dt
                self.threshold = threshold
                self.V_init = V_init
                self.VL = VL
                self.V_reset = V_reset
                self.gl = gl
                self.Cm = Cm
                #self.delta_trace = list()
                self.N = 0
                self.spike_gradient = ATan.apply

                self.layer_size = size

                
                self.trace_builder = compute.trace_builder(size, alpha)
                
                # precompute 2000 weight values
                if len(FLIF.weight_vector) == 0:
                        nv = np.arange(1999)
                        FLIF.weight_vector = (2001-nv)**(1-self.alpha)-(2000-nv)**(1-self.alpha)
        

        # Both I and V_old are (batch_size)x(layer size) tensors; need to adapt function
        def forward(self, I_old, V_old):

                I = I_old #+ 3
                N = self.N


                if N == 0:
                        V_new = torch.ones_like(V_old)*self.V_init
                        spike = torch.zeros_like(V_old)
                        self.N += 1

                        return spike, V_new

                
                spike = self.spike_gradient((V_old - self.threshold))

               

                        
		# Build weight vector if needed
                # For current version, this is unnecessary
                """
                w_available = len(FLIF.weight_vector)
                if N == w_available:

                        # largest item is (N+1) - N

                        missing = N - w_available

                        new_weights = list()
                        for i in range(N+1000, N, -1):

                                weight = (i+1)**(1-self.alpha) - (i)**(1-self.alpha)
                                new_weights.append(weight)

                        FLIF.weight_vector = np.concatenate(new_weights, FLIF.weight_vector)
                """
                
                        
                if N == 1:
			#Classic LIF
                        tau = self.Cm / self.gl
                        V_new = V_old + (self.dt/tau)*(-1 * V_old + I/self.gl) # new 1x(layer size) tensor

                        #self.V_trace.append(V_new)

                        #self.delta_trace.append(torch.sub(V_new, V_old))
                        self.trace_builder.init_trace(V_new, V_old)

                else:
			#Fractional LIF
                        
                        V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(V_old-self.VL)+I) / self.Cm + V_old


                        # Compute memory trace
                        """
                        delta_trace = torch.stack(self.delta_trace).detach()
                        delta_trace = torch.transpose(delta_trace, 2, 0)
                        delta_trace = torch.transpose(delta_trace, 0, 1)
                        memory_V = torch.matmul(delta_trace, torch.tensor(FLIF.weight_vector[-N+1:]).float())
                        """
                        memory_V = self.trace_builder.get_memory_trace()
                        
                        V_new = torch.sub(V_new, memory_V)

                        #self.delta_trace.append(torch.sub(V_new, V_old))
                        self.trace_builder.update_trace(V_new, V_old)

                
                reset = (spike * (self.threshold - self.V_reset)).detach()
                #reset = (spike.mul(V_new - self.V_reset)).detach()
                                        
                V_new = torch.sub(V_new, reset)
                self.N += 1
                        
                return spike, V_new

        def init_mem(self, batch_size):
                #self.delta_trace.clear()
                self.trace_builder.reset_trace()
                self.N = 0
                return torch.zeros(batch_size, self.layer_size)
                
        

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
