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
import random

mem_stuff = list()

# Standard LIF model; takes in a voltage value and calculates
# the voltage at the next time step.
def leaky_integrate_neuron(V, time_step=0.1, I=4, gl=0.025, Cm=0.5):
        tau = Cm / gl
        V = V + (time_step/tau)*(-1 * V + I/gl)
        return V

# Fractional LIF model; uses previous voltage values to approximate next voltage
def frac_num_lif(V_trace, V_weight, I=4, thresh=-50, V_reset=-70,  Vl=-70, dt=0.1, beta=0.2, gl=0.025, Cm=0.5):

        N = len(V_trace)
        V = V_trace[N - 1]
        tau = Cm / gl


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
        spike = (V_new > thresh)
        V_new -= (V_new - V_reset)*spike

        return V_new


# MAIN FUNCTION
def main():

        num_steps = 250
        beta = 0.2
        # initial voltage, -70 millivolts
        V = -70
        V_trace = list() # list captures all computed voltage values
        nv = np.arange(num_steps-1)
        V_weight = (num_steps-nv+1)**(1-beta)-(num_steps-nv)**(1-beta)

        I = 10

        inputs = list()

        for i in range(num_steps):

                inputs.append(random.randint(3, 7))

                
        
        
        for step in range(num_steps):

                I = inputs[step]

		# First 2 time steps t0, t1 are computed using regular LIF
                if step < 2:

                        V_trace.append(V)
                        V = leaky_integrate_neuron(V, I=I)
                        mem_stuff.append(0)

		# All others use the fractional
                else:

                        
                        V_trace.append(V)
                        V = frac_num_lif(V_trace, V_weight, I=I)
                        #print(V)
                        


        # slope processing
        trace_slope = list()
        mem_slope = list()
        differential = list()
        for step in range(num_steps-1):

                trace_slope.append(V_trace[step+1] - V_trace[step])
                mem_slope.append(mem_stuff[step+1] - mem_stuff[step])

                if mem_slope[step] == 0:
                        differential.append(0)
                else:
                        differential.append(trace_slope[step] - mem_slope[step])
        

                        
	#Plotting the list of voltage values
        #plotsrc.plot_mem(V_trace, 0, num_steps, -0.071, -0.048, "Fractional Leaky Neuron Model", True, str(datetime.datetime.now()))
        """
        figure, ax = plt.subplots(3)
        ax[0].plot(np.arange(num_steps-1), trace_slope)
        ax[1].plot(np.arange(num_steps-1), mem_slope)
        ax[2].plot(mem_slope, trace_slope)
        ax[0].set_title("Neuron Voltage")
        ax[1].set_title("Memory Trace")
        ax[2].set_title("?")
        plt.show()
        """
        plt.plot(np.arange(num_steps-1), trace_slope, label = "T")
        plt.plot(np.arange(num_steps-1), mem_slope, label = "M")
        plt.legend()
        plt.show()

        
if __name__ == '__main__':
        main()
