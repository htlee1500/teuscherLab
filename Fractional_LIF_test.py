import math
import numpy as np

class Frac_Test():
    
    def __init__(self, num_steps):

        self.V_trace = list()
        self.N = 0

        nv = np.arange(num_steps - 1)
        self.weights = (num_steps -nv+1)**(0.8) - (num_steps -nv)**(0.8)

    def run(self, I, thresh=-50, V_reset=-70,  Vl=-70, dt=0.1, beta=0.2, gl=0.025, Cm=0.5):

        N = self.N


        if N == 0:
            self.V_trace.append(V_reset)
            self.N += 1
            return V_reset

        elif N == 1:
            V = self.V_trace[N - 1]
            tau = Cm / gl
            V_new = V + (0.1/tau)*(-1*V + I/gl)
            
            
            spike = (V >= thresh)
            V_new -= (V_new - V_reset)*spike
            self.V_trace.append(V_new)
            self.N += 1
            return V_new

        else:
            
            tau = Cm / gl
            V = self.V_trace[N-1]

            V_new = dt**(beta) * math.gamma(2-beta) * (-gl*(V-Vl)+I) / Cm + V
            
	    # Computing voltage trace
            delta_V = np.subtract(self.V_trace[1:],self.V_trace[0:(N-1)])
            #delta_V = delta_V[0:N-2]
            memory_V = np.inner(self.weights[-len(delta_V):], delta_V)#np.inner(V_weight[-len(delta_V):],delta_V)
            #print("Free neuron", memory_V)
            V_new -= memory_V

	    # Reset voltage if spiking (not sure if this is computationally efficient)
            spike = (V >= thresh)
            V_new -= (V_new - V_reset)*spike
            self.V_trace.append(V_new)
            self.N += 1
            return V_new
