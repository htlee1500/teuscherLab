
# WARNING: Do not use. This is just a dump for me to put stuff that I'm not using, but am not ready to delete.

                I = I_old
                N = self.trace_len
                if N == 0:
                        V_new = torch.ones_like(I) *-70
                        self.add_to_trace(V_new)
                        return torch.zeros_like(V_new), V_new
                        
                
                spike = self.spike_gradient((V_old - self.threshold))
                reset = (spike * (self.threshold - self.V_reset)).detach()
                V_new = 0
               
		
                        
                # Build weight vector if needed
                w_available = len(FLIF.weight_vector)
                if N > w_available:

                        missing = N - w_available

                        for i in range(w_available, N):

                                weight = i**(1-self.alpha) - (i-1)**(1-self.alpha)
                                FLIF.weight_vector.append(weight)

                if N < 2:
                        
			#Classic LIF
                        V_new = V_old + (self.dt/self.tau)*(-1 * V_old + I/self.gl) # new 1x(layer size) tensor
                        #print(self.trace_len)
                        self.add_to_trace(V_new)

                else:
			#Fractional LIF
                        V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(V_old-self.VL)+I) / self.Cm + V_old

                        """
                        # Compute memory trace
                        # V_trace is a file of all previous values, with dimensions num steps x layer size
                        trace = self.get_trace_matrix()
                        delta_trace = np.subtract(trace[1:], trace[0:N-1])
                        delta_trace = np.transpose(delta_trace) # transpose to make delta_trace a layer size x num steps matrix
                        
                        
                        memory_V = np.dot(delta_trace, FLIF.weight_vector[-len(trace)+1:]) # memory_V is a 1 x layer_size vector
                        memory_tensor = torch.tensor(memory_V)
                        V_new = torch.sub(V_new, memory_tensor)
                        self.add_to_trace(V_new)
                        """
                        
                        # Compute memory trace
                        # voltage_trace is a 3d matrix of all previous values (trace_len x batch_size x layer_size)
                        dims = V_old.size()
                        voltage_trace = self.get_trace_matrix(dims[0])
                        delta_trace = np.subtract(voltage_trace[1:], voltage_trace[0:N-1])
                        delta_trace = np.transpose(delta_trace, (1, 2, 0)) # transpose to make delta_trace batch_size x layer_size x trace_len

                        memory_V = np.dot(delta_trace, FLIF.weight_vector[-self.trace_len+1:])
                        memory_tensor = torch.tensor(memory_V)

                        V_new = torch.sub(V_new, memory_tensor)
                        self.add_to_trace(V_new)

            V_new = torch.sub(V_new, reset)
                return spike, V_new

        # Returns num steps x batch size x layer size matrix of voltage values
        def get_trace_matrix(self, batch_size):
                matrix_cube = list()
                matrix_face = list()

                with open(self.V_trace, "r") as reader:
                        for i, line in enumerate(reader):

                                if i > 0 and i % batch_size == 0:

                                        matrix_cube.append(matrix_face)
                                        matrix_face.clear()
                                
                                
                                matrix_face.append(self.parse_line(line))

                        matrix_cube.append(matrix_face)
                                
                        reader.close()

                return matrix_cube

        # Gets latest trace values from storage
        # get last [batch_size] lines from the trace, convert to batch size x layer size matrix
        def get_latest_trace(self, batch_size):
                
                first_line = batch_size * (self.trace_len - 1)
                latest_trace = list()
                
                with open(self.V_trace, "r") as reader:
                        for i, line in enumerate(reader):

                                if i >= first_line:
                                        latest_trace.append(self.parse_line(line))


                reader.close()

                latest_trace = torch.tensor(latest_trace)
                print(latest_trace.size())

                return latest_trace
                	
                	
        # Turns string of comma-separated values into a list
        def parse_line(self, string):
                parsed = list()
                value_buffer = ""
                for char in string:
                        if ord(char) == 44: # comma detected
                                parsed.append(float(value_buffer))
                                value_buffer = ""
                        elif ord(char) == 10: # newline char detected
                                parsed.append(float(value_buffer))
                                value_buffer = ""
                        else:
                                value_buffer += char
                return parsed

        def trace_to_line(self, trace):
                string_version = ""
                
                for voltage in trace:
                        
                        string_version += (str(voltage))
                        string_version += ","
                        
                string_version = string_version[0:len(string_version)-1]
                string_version += "\n"
                return string_version
        	
        
        # Adds the next set of voltage values to the storage file (trace is a batch size x layer size tensor)
        def add_to_trace(self, trace):
                self.trace_len += 1
                list_version = trace.tolist() # 2D list
                with open(self.V_trace, "a") as writer:

                        for row in list_version:
                        
                                writer.write(self.trace_to_line(row))
                
                        writer.close()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""

                        N = self.trace_len
                        
                        
                        # Generate more weight values if necessary
                        if N == len(FLIF.weight_vector):
                                print("Generating additional weights...")
                                new_vals = list()
                                for i in range(1, 1000):
                                        index = N + i
                                        weight = (index + 1)**(1-self.alpha) - index**(1-self.alpha)
                                        new_vals.append(weight)

                                FLIF.weight_vector = np.concatenate((new_vals, FLIF.weight_vector), axis = 0)

                        

                        match N:
                                case 0:
                                        V_new = torch.ones_like(I)*-70
                                        spike = self.spike_gradient((V_new - self.threshold))
                                        reset = (spike * (self.threshold - self.V_reset)).detach()

                                        V_new = torch.sub(V_new, reset)
                                        V_new_all.append(V_new)
                                        spike_all.append(spike)

                                        self.V = V_new
                                        self.trace_len += 1
                                        continue
                                case 1:
                                        V_new = self.V + (self.dt/self.tau)*(-1 * self.V + I/self.gl)
                                        
                                case _:
                                        V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(self.V-self.VL)+I) / self.Cm + self.V


                                        # Compute memory trace
                                        #memory_V = np.dot(FLIF.weight_vector[-N+1:], delta_trace)
                                        memory_V = np.matmul(FLIF.weight_vector[-N+1:], self.delta_trace)

                                        memory_tensor = torch.tensor(memory_V, dtype=torch.float64)
                                        V_new = torch.sub(V_new, memory_tensor)
                                        
                        spike = self.spike_gradient((self.V - self.threshold))
                        reset = (spike * (self.threshold - self.V_reset)).detach()
                        
                        V_new = torch.sub(V_new, reset)
                        delta = torch.sub(V_new, self.V)
                        self.delta_trace.append(delta.tolist())
                        self.V = V_new
                        self.trace_len += 1
                        V_new_all.append(V_new)
                        spike_all.append(spike)



                         
                num_output = len(I_sequence)
                
                spikes = [0]*num_output
                trace = [0]*num_output
                delta_trace = self.delta_trace.get(index)
                V = 0

                N = self.N_vector[index]
                
                for I in I_sequence:

                        match N:
                                case 0:
                                        V_new = torch.ones(1)*-70
                                        spike = self.spike_gradient((V_new - self.threshold))
                                        reset = (spike * (self.threshold - self.V_reset)).detach()

                                        V_new = torch.sub(V_new, reset)
                                        trace[N] = V_new[0].item()
                                        spikes[N] = spike[0].item()

                                        V = V_new
                                        N += 1
                                        continue

                                case 1:
                                
                                       V_new = V + (self.dt/self.tau)*(-1 * V + I/self.gl)

                                case _:
 
                                       V_new = self.dt**(self.alpha) * math.gamma(2-self.alpha) * (-self.gl*(V-self.VL)+I) / self.Cm + V

                                       memory_V = np.inner(FLIF.weight_vector[-N+1:], delta_trace)
                                       memory_tensor = torch.tensor(memory_V)

                                       V_new = torch.sub(V_new, memory_tensor)

                        spike = self.spike_gradient(V - self.threshold)
                        reset = (spike * (self.threshold - self.V_reset)).detach()

                        V_new = torch.sub(V_new, reset)
                        delta  = torch.sub(V_new, V)
                        delta_trace.append(delta[0].item())
                        V = V_new
                        trace[N] = V_new[0].item()
                        spikes[N] = spike[0].item()
                        N += 1

                N -= 1
                
                if (index+1) % 50 == 0:
                        print("Calculations for neuron ", (index+1), " completed.")

                
                        
               
               
                return (index, spikes, trace, delta_trace, V, N)
