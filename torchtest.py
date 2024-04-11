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

def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
	tau = R*C
	U = U + (time_step/tau)*(-U+ I*R)
	return U

# Note: will need to adapt for more complex I functions
def fractional_lif_neuron(V, Vl, V_init=0, t, time_step=1e-3, I=1, dI=0, gl=2.5e-8, C=5e-10, g_single, g_double):

	tau = C / gl
	V = V + 1/gl * dI + (V_init - Vl - I/gl)*compute.compute_ML_double(0.2, tau, t, 100, g_double)*B*


def main():
"""
	num_steps = 100
	U = 0.9
	U_trace = []  # keeps a record of U for plotting

	for step in range(num_steps):
		U_trace.append(U)
		U = leaky_integrate_neuron(U)  # solve next step of U

	plotsrc.plot_mem(U_trace, "Leaky Neuron Model")
	print("plotted")
"""

	

if __name__ == '__main__':
	main()
