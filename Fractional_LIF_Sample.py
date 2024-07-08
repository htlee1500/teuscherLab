# imports
import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import plotsrc
import compute
import math
import datetime

mem_stuff = list()

# Standard LIF model; takes in a voltage value and calculates
# the voltage at the next time step.
def leaky_integrate_neuron(V, time_step=0.1, I=4, gl=0.025, Cm=0.5):
        tau = Cm / gl
        V = V + (time_step/tau)*(-1 * V + I/gl)
        return V

# Fractional LIF model; uses previous voltage values to approximate next voltage
def frac_num_lif(V_trace, V_weight, I=4, thresh=-65, V_reset=-70,  Vl=-70, dt=0.1, beta=0.15, gl=0.025, Cm=0.5):

        N = len(V_trace)
        V = V_trace[N - 1]
        tau = Cm / gl

        spike = (V > thresh)

	# V_new is the voltage at t_N+1
	# Computing Markov term
        V_new = dt**(beta) * math.gamma(2-beta) * (-gl*(V-Vl)+I) / Cm + V

	# Computing voltage trace
        delta_V = np.subtract(V_trace[1:],V_trace[0:(N-1)])
        #delta_V = delta_V[0:N-2]
        memory_V = np.inner(V_weight[-len(delta_V):], delta_V)#np.inner(V_weight[-len(delta_V):],delta_V)

        mem_stuff.append(memory_V)

        V_new -= memory_V

	# Reset voltage if spiking (not sure if this is computationally efficient)
        V_new -= (V_new - V_reset)*spike

        return V_new


# MAIN FUNCTION
def main():

        num_steps = 50
        beta = 0.15
        # initial voltage, -70 millivolts
        V = -70
        V_trace = list() # list captures all computed voltage values
        nv = np.arange(num_steps-1)
        V_weight = (num_steps-nv+1)**(1-beta)-(num_steps-nv)**(1-beta)

        I = list()

        val = 3.5
        nextval = 1.5

        for i in range(num_steps):

                if i < 200:
                        temp = val
                        val = nextval
                        nextval = temp

                I.append(val)
                
        
        
        for step in range(int(num_steps)):

		# First 2 time steps t0, t1 are computed using regular LIF
                if step < 2:

                        V_trace.append(V)
                        V = leaky_integrate_neuron(V, I=I[i])

		# All others use the fractional
                else:

                        
                        V_trace.append(V)
                        V = frac_num_lif(V_trace, V_weight, I=I[i])
                        #print(V)
                        

        for step in range(int(num_steps)):

                V_trace[step] = V_trace[step]*1e-3

                
	#Plotting the list of voltage values
        plotsrc.plot_mem(V_trace, 0, num_steps, -0.071, -0.048, "Fractional Leaky Neuron Model", True, str(datetime.datetime.now()))

        #plt.plot(np.arange(num_steps - 2), mem_stuff)
        #plt.show()


if __name__ == '__main__':
        main()
