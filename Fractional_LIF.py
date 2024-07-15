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

        def __init__(self, size, device, num_steps, alpha=0.2, dt=0.1, threshold=-50, V_init=-70, VL=-70, V_reset=-70, gl=0.025, Cm=0.5):

                super().__init__() # readd FLIF, self to constructor if needed.

                self.alpha = alpha
                self.dt = dt
                self.threshold = threshold
                self.V_init = V_init
                self.VL = VL
                self.V_reset = V_reset
                self.gl = gl
                self.Cm = Cm
                self.delta_trace = torch.zeros(0)
                self.N = 0
                self.spike_gradient = ATan.apply
                self.device = device
                self.layer_size = size
                self.num_steps = num_steps

                
                # precompute x weight values
                if len(FLIF.weight_vector) == 0:
                        x = num_steps
                        
                        nv = np.arange(x-1)
                        FLIF.weight_vector = torch.tensor((x+1-nv)**(1-self.alpha)-(x-nv)**(1-self.alpha)).float().to(self.device)
        

        # Both I and V_old are (batch_size)x(layer size) tensors
        def forward(self, I_old, V_old):

                I = I_old #+ 3
                N = self.N


                if N == 0:
                        V_new = (torch.ones_like(V_old)*self.V_init).to(self.device)
                        spike = torch.zeros_like(V_old).to(self.device)
                        self.N += 1

                        return spike, V_new

                
                spike = self.spike_gradient((V_old - self.threshold))

               

                
                        
                if N == 1:
			#Classic LIF
                        tau = self.Cm / self.gl
                        V_new = V_old + (self.dt/tau)*(-1 * V_old + I/self.gl) # new 1x(layer size) tensor
                        

                else:
			#Fractional LIF
                        
                        V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(V_old-self.VL)+I) / self.Cm + V_old


                        # Compute memory trace
                        delta_trace = self.delta_trace[:,:,0:N-1]

                        weights = FLIF.weight_vector[-N+1:]
                        memory_V = torch.matmul(delta_trace, weights)

                        V_new = torch.sub(V_new, memory_V)
                        
 
                spike = self.spike_gradient((V_old - self.threshold))
                
                reset = (spike * (V_new - self.V_reset)).detach()
                #reset = (spike.mul(V_new - self.V_reset)).detach()
                                        
                V_new = torch.sub(V_new, reset)
                self.update_delta(V_new, V_old)
                self.N += 1

                return spike, V_new

        def init_mem(self, batch_size):
                self.delta_trace = torch.zeros(0)
                #self.trace_builder.reset_trace()
                self.N = 0
                return torch.zeros(batch_size, self.layer_size).to(self.device)

        def update_delta(self, V_new, V_old):
                # batch layer step

                delta = torch.sub(V_new, V_old).detach()
                
                if self.N == 1:
                        # init delta_trace
                        batch_size = delta.size(0)
                        self.delta_trace = torch.zeros(batch_size, self.layer_size, self.num_steps).to(self.device)

                        
                
                self.delta_trace[:,:,self.N-1] = delta
                
        

class ATan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, mem):
                spk = (mem >= 0).float() # Heaviside on the forward pass: Eq(2)
                ctx.save_for_backward(mem)  # store the membrane for use in the backward pass
                return spk

        @staticmethod
        def backward(ctx, grad_output):
                (mem, ) = ctx.saved_tensors  # retrieve the membrane potential
                grad = 1 / (1 + (np.pi * mem).pow_(2)) * grad_output # Eqn 5
                return grad
