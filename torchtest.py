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

def leaky_integrate_neuron(V, time_step=1e-4, I=2.5e-9, gl=2.5e-8, Cm=5e-10):
	tau = Cm / gl
	V = V + (time_step/tau)*(-1 * V + I/gl)
	return V

"""
def fractional_lif_neuron(V, Vl, V_init=0, t, time_step=1e-3, I=1, dI=0, gl=2.5e-8, C=5e-10, g_single, g_double):

	#tau = C / gl
	#V = V + 1/gl * dI + (V_init - Vl - I/gl)*compute.compute_ML_double(0.2, tau, t, 100, g_double)*B*

"""
def frac_num_lif(V, V_trace, Vl=-7e-2, dt=1e-4, beta=0.2, gl=2.5e-8, Cm=5e-10,I=0.3e-9):

	N = len(V_trace)
	tau = Cm / gl

	if N == 1:
		#V_new = Vl + I/gl + (V_trace[0] - Vl - I/gl)*compute.mittag_leffler(beta, tau, time_step)
		#return V_new
		return leaky_integrate_neuron(V)

	#markov_term = (dt)**beta * math.gamma(2-beta) * (-1 * gl * (V-Vl) + I) / Cm + V


	voltage_trace = 0

	for k in range(0, N-1):

		delta_V = V_trace[k+1] - V_trace[k]
		weight_V = (N-k)**(1-beta) - (N - 1 - k)**(1-beta)

		voltage_trace += delta_V * weight_V
	"""
	print("Step " + str(N))
	print("Markov: " + str(markov_term))
	print("Trace: " + str(voltage_trace))
	print("--------------------------------------")
	"""
	#V_new = markov_term - voltage_trace

	gamma = math.gamma(2-beta)
	V_new = dt**(-1 * beta) / gamma * (voltage_trace - V)
	V_new -= (gl*Vl + I)/Cm
	V_new = V_new / (-1 * gl / Cm - dt**(-1 * beta) / gamma)

	return V_new

def main():

	num_steps = 1000
	dt = 1e-4

	V = -7e-2
	V_trace = list()

	for step in range(int(num_steps)):
		#print(V)
		V_trace.append(V)
		#print(str(V) + " -> ")
		V = frac_num_lif(V, V_trace)
		#V = leaky_integrate_neuron(V)
		#print(V)

	f = open("data.txt", "w")
	now = datetime.datetime.now()
	f.write(str(now) + "\n")
	for step in range(int(num_steps)):
		f.write(str(V_trace[step]) + "\n")

	f.close()

	plotsrc.plot_mem(V_trace, 0, num_steps, -0.07, -0.06, "Fractional Leaky Neuron Model", False, str(now))


if __name__ == '__main__':
	main()
