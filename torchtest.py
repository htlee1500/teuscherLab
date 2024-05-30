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

# Standard LIF model; takes in a voltage value and calculates
# the voltage at the next time step.
def leaky_integrate_neuron(V, time_step=1e-4, I=0.3e-9, gl=2.5e-8, Cm=5e-10):
	tau = Cm / gl
	V = V + (time_step/tau)*(-1 * V + I/gl)
	return V

# Fractional LIF model; uses previous voltage values to approximate next voltage
def frac_num_lif(V, V_trace, Vl=-7e-2, dt=1e-4, beta=0.2, gl=2.5e-8, Cm=5e-10,I=0.3e-9):

	N = len(V_trace)
	tau = Cm / gl

	# V_new is the voltage at t_N+1
	# Computing Markov term
	V_new = dt**(beta) * math.gamma(2-beta) * (-gl*(V-Vl)+I) / Cm + V

	# Computing voltage trace

	voltage_trace = 0
	# Loop calculates terms of the voltage trace sum
	for k in range(1, N):

		delta_V = V_trace[k] - V_trace[k-1]
		weight_V = (N+1-k)**(1-beta) - (N-k)**(1-beta)

		term = delta_V * weight_V
		voltage_trace += term

	V_new -= voltage_trace
	return V_new

# MAIN FUNCTION
def main():

	num_steps = 2000

	# initial voltage, -70 millivolts
	V = -7e-2
	V_trace = list() # list captures all computed voltage values

	for step in range(int(num_steps)):

		# First 3 time steps t0, t1, t2 are computed using regular LIF
		if step < 3:

			V_trace.append(V)
			V = leaky_integrate_neuron(V)

		# All others use the fractional LIF
		else:
			V_trace.append(V)
			V = frac_num_lif(V, V_trace)

	# This block writes the voltage values to a text file
	# for troubleshooting and debugging
	f = open("data.txt", "w")
	now = datetime.datetime.now()
	f.write(str(now) + "\n")
	for step in range(int(num_steps)):
		f.write(str(V_trace[step]) + "\n")

	f.close()

	# Plotting the list of voltage values
	plotsrc.plot_mem(V_trace, 0, num_steps, -0.07, -0.05, "Fractional Leaky Neuron Model", True, str(now))


if __name__ == '__main__':
	main()
