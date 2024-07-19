
# WARNING: Do not use. This is just a dump for me to put stuff that I'm not using, but am not ready to delete.

                 for item in range(batch):

                        item_current_trace = list()
                        item_spike_trace = list()

                        hidden_mem = self.hidden_layer.init_mem()
                        output_mem = self.output_layer.init_mem()

                        hidden_spikes = torch.zeros_like(hidden_mem)
                        output_spikes = torch.zeros_like(output_mem)

                        self.reset_fire_rates()
                        """
                        torch.cuda.empty_cache()
                        t = torch.cuda.get_device_properties(0).total_memory
                        r = torch.cuda.memory_reserved(0)
                        a = torch.cuda.memory_allocated(0)
                        print(f"For Item {item}: Free mem: {(r-a)/1000000}MB. Allocated: {a/1000000}MB. Mem remaining: {(t-a)/1000000}MB")
                        """
                        
                        for step in range(self.num_steps):

                                #print(f"Made it to ({item}, {step})")
                                
                                input_spikes = data[item, step]
                        
                                # compute current to hidden layer:
                                # forward from input
                                # feedback from output
                                # lateral connections
                                hidden_current = self.forward_hidden(input_spikes)
                                hidden_current += self.feedback(output_spikes)
                                hidden_current += self.lateral_hidden(hidden_spikes)

                                # Stimulate hidden layer:
                                hidden_spikes, hidden_mem = self.hidden_layer(hidden_current, hidden_mem)


                                # compute current to output layer:
                                # forward from hidden
                                # lateral connections
                                # supervised current from targets (if training = True)
                                output_current = self.forward_output(hidden_spikes)
                                output_current += self.lateral_output(output_spikes)

                                output_current += input_vector[item]

                                # Stimulate output layer:
                                output_spikes, output_current = self.output_layer(output_current, output_mem)

                                item_current_trace.append(output_current)
                                item_spike_trace.append(output_spikes)
                        

                                # compute updates to a's and thetas
                                if self.no_count:
                                        self.input_a = input_spikes
                                        self.input_theta = input_spikes
                                        
                                        self.hidden_a = hidden_spikes
                                        self.hidden_theta = hidden_spikes

                                        self.output_a = output_spikes
                                        self.output_theta = output_spikes

                                        self.no_count = True

                                else:

                                        self.input_a = (1 - self.dt/tau_a) * self.input_a + input_spikes / tau_a
                                        self.input_theta = (1 - self.dt/tau_theta) * self.input_theta + input_spikes / tau_theta
                                        
                                        self.hidden_a = (1 - self.dt/tau_a) * self.hidden_a + hidden_spikes / tau_a
                                        self.hidden_theta = (1 - self.dt/tau_theta) * self.hidden_theta + hidden_spikes / tau_theta

                                        self.output_a = (1 - self.dt/tau_a) * self.output_a + output_spikes / tau_a
                                        self.output_theta = (1 - self.dt/tau_theta) * self.output_theta + output_spikes / tau_theta

                                        
                                # compute weight updates:
                                in_diff = self.input_a - self.input_theta
                                hid_diff = self.hidden_a - self.hidden_theta
                                out_diff = self.output_a - self.output_theta
                                
                                # forward weights
                                weight = self.forward_hidden.weight # hidden x input
                                
                                delta_weight = self.alpha * torch.mul(
                                        torch.broadcast_to(in_diff, (num_hidden, num_input)),
                                        torch.transpose(torch.broadcast_to(hidden_spikes, (num_input, num_hidden)), 0, 1))
                                delta_weight -= weight/tau

                                weight = self.dt*delta_weight + weight
                                self.forward_hidden.weight = nn.Parameter(weight)

                                


                                weight = self.forward_output.weight # output x hidden
                                delta_weight = self.alpha * torch.mul(
                                        torch.broadcast_to(hid_diff, (num_output, num_hidden)),
                                        torch.transpose(torch.broadcast_to(output_spikes, (num_hidden, num_output)), 0, 1))
                                delta_weight -= weight/tau

                                weight = self.dt*delta_weight + weight
                                self.forward_output.weight = nn.Parameter(weight)

                                
                                # feedback weights
                                weight = self.feedback.weight
                                delta_weight = self.beta * torch.mul(
                                        torch.transpose(torch.broadcast_to(hid_diff, (num_output, num_hidden)), 0, 1),
                                        torch.broadcast_to(output_spikes, (num_hidden, num_output)))
                                delta_weight -= weight/tau

                                weight = self.dt*delta_weight + weight
                                self.feedback.weight = nn.Parameter(weight)

                                
                                # lateral weights
                                weight = self.lateral_output.weight
                                delta_weight = -1 * self.gamma * torch.mul(
                                        torch.broadcast_to(out_diff, (num_output, num_output)),
                                        torch.broadcast_to(output_spikes, (num_output, num_output)))
                                delta_weight -= weight/tau

                                weight = self.dt*delta_weight + weight
                                weight.fill_diagonal_(0)
                                self.lateral_output.weight = nn.Parameter(weight)



                                weight = self.lateral_hidden.weight
                                delta_weight = self.eta * torch.mul(
                                        torch.broadcast_to(hid_diff, (num_hidden, num_hidden)),
                                        torch.broadcast_to(hidden_spikes, (num_hidden, num_hidden)))
                                delta_weight -= weight/tau
                                weight = self.dt*delta_weight + weight
                                weight.fill_diagonal_(0)
                                self.lateral_hidden.weight = nn.Parameter(weight)



                        output_current_trace.append(torch.stack(item_current_trace))
                        output_spike_trace.append(torch.stack(item_spike_trace))

                return torch.transpose(torch.stack(output_spike_trace), 1, 0), torch.transpose(torch.stack(output_current_trace), 1, 0)
