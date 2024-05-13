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

def leaky_integrate_neuron(V, time_step=1e-4, I=0.3e-9, gl=2.5e-8, Cm=5e-10):
	tau = Cm / gl
	V = V + (time_step/tau)*(-1 * V + I/gl)
	return V

def frac_num_lif(V, V_trace, Vl=-7e-2, dt=1e-4, beta=0.2, gl=2.5e-8, Cm=5e-10,I=0.3e-9):
	
	N = len(V_trace)
	tau = Cm / gl

	if N == 1:
		V_new = Vl + I/gl + (V - Vl - I/gl)*compute.mittag_leffler()
		return V_new
		#return leaky_integrate_neuron(V)

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
	gamma = math.gamma(2-beta)
	V_new = (gl*Vl + I)/Cm
	V_new = V_new * gamma * (dt**beta)
	V_new = V_new + V - voltage_trace
	V_new = V_new / (1 + gamma*(dt**beta)*(gl/Cm))
	return V_new

def frac_test_neuron(V, V_trace, dt=1e-4, beta=0.2):

	N = len(V_trace)

	if N == 1:
		V_new = -0.014957788292486045
		return V_new

	trace = 0
	for k in range(0, N-1):
		delta_V = V_trace[k+1] - V_trace[k]
		weight_V = (N-k)**(1-beta) - (N - 1 - k)**(1-beta)

		trace += delta_V*weight_V

	V_new = dt**(-1*beta)/math.gamma(2-beta) * (trace - V)
	V_new -= 1

	V_new = V_new / (-1 - (dt**(-1*beta)/math.gamma(2-beta)))


	return V_new


def main():

	num_steps = 2000
	"""
	V = -7e-2
	V_trace = list()
	V_trace_adapted = list()
	#U = -7e-2
	#U_trace = list()

	for step in range(int(num_steps)):

		V_trace.append(V)
		V_trace_adapted.append(V*-1)
		V = frac_num_lif(V, V_trace)

		#U_trace.append(U)
		#U = leaky_integrate_neuron(U)

	f = open("data.txt", "w")
	now = datetime.datetime.now()
	f.write(str(now) + "\n")
	for step in range(int(num_steps)):
		f.write(str(V_trace_adapted[step]) + "\n")

	f.close()

	plotsrc.plot_mem(V_trace_adapted, 0, num_steps, -0.07, -0.05, "Fractional Leaky Neuron Model", True, str(now))
	"""


	U = 0
	U_trace = list()

	for step in range(num_steps):
		print(U)
		U_trace.append(U)
		U = frac_test_neuron(U, U_trace)
	now = datetime.datetime.now()
	plotsrc.plot_mem(U_trace, 0, num_steps, 0, 0.7, "Sanity Test", True, str(now))

if __name__ == '__main__':
	main()
