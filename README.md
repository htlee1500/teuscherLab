# teuscherLab
Repo for work in the lab of Prof. Christof Teuscher


Executables:

* snn_rate.py - trains a spiking neural network with fractional-order dynamics on MNIST using rate-encoded inputs

* plot_builder.py - script to plot membrane voltage traces and spike raster plots of trained SNN

* snn_dead.py - performs robustness testing on SNN by removing some inputs

* spiked_data_builder.py - generates and saves a spiked testing set on MNIST for snn_dead testing

* snn_noisy.py - performs robustness testing on SNN by adding noise to inputs

* data_builder.py - generates and saves a testing set on MNIST for snn_noisy testing

* statreader.py - reads cProfile data from bottleneck.txt

* Fractional_LIF_Sample.py - shows membrane voltage trace of a single neuron with fractional leaky-integrate-and-fire dynamics