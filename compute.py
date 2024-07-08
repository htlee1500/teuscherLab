import torch
import numpy as np

import ray

# trace_builder object stores and updates the trace
# Also computes memory trace values for fractional neurons
class trace_builder():

        weight_vector = np.empty(1)

        def __init__(self, layer_size, alpha):

                self.delta_trace = list() # (num_steps - 1) batch_size x layer_size tensors
                self.N = 0 # trace length
                self.size = layer_size

                if trace_builder.weight_vector.size == 1:

                        nv = np.arange(1999)
                        trace_builder.weight_vector = (2001-nv)**(1-alpha) - (2000-nv)**(1-alpha)

        def init_trace(self, V_new, V_old):
                # batch_size x layer_size tensors as input
                V_delta = torch.sub(V_new, V_old)#.detach()
                #V_delta = torch.transpose(V_delta, 0, 1)

                self.delta_trace.append(V_delta)
                
                self.N = 1

        def reset_trace(self):

                self.delta_trace = list()
                self.N = 0

        # Given new voltage values; update the traces and compute the memory trace
        # return in tensor form

        def update_trace(self, V_new, V_old):
                # add incoming V_new values
                # V_new is a tensor with dims batch_size x layer_size

                V_delta = torch.sub(V_new, V_old)#.detach()
                #V_delta = torch.transpose(V_delta, 0, 1)

                self.delta_trace.append(V_delta)
                
                self.N += 1

                        
        def get_memory_trace(self):
        
                """
                if self.N > 0:
                        ray.init(include_dashboard = False, configure_logging = True, logging_level = "error")
                        N = ray.put(self.N)
                        weights = ray.put(trace_builder.weight_vector)
                        delta = ray.put(self.delta_trace)

                        mem_trace_list = ray.get([trace_builder.step_trace.remote(i, weights, delta) for i in range(self.N)])

                        ray.shutdown()
                else:

                        mem_trace_list = [0]*self.N
                        for i in range(self.N):
                                mem_trace_list[i] = torch.mul(self.delta_trace[i], trace_builder.weight_vector[-i-1])
                

                memory_trace = mem_trace_list[0]

                for i in range(1, self.N):

                        memory_trace = torch.add(memory_trace, mem_trace_list[i])

                return memory_trace
                """

                delta = torch.stack(self.delta_trace)
                delta = torch.transpose(delta, 2, 0) # now layer x batch x steps
                delta = torch.transpose(delta, 0, 1) # now batch x layer x steps

                if self.N > 10000:

                        ray.init(include_dashboard = False, configure_logging = True, logging_level = "error")
                        weights = ray.put(torch.tensor(trace_builder.weight_vector[-self.N:]).float())
                        delta_ref = ray.put(delta)
                        batch = delta.size()[0]


                        memory_V = ray.get([trace_builder.single_sample.remote(i, weights, delta_ref) for i in range(batch)])

                        memory_V = torch.stack(memory_V)

                        ray.shutdown()

                else:

                        memory_V = torch.matmul(delta, torch.tensor(trace_builder.weight_vector[-self.N:]).float())

                return memory_V


        @ray.remote
        def single_sample(index, weights, delta):

                my_delta = delta[index] # layer x steps

                my_mem = torch.matmul(my_delta, weights) # layer x 1

                return my_mem
                
