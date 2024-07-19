import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm
import datetime


def test_system(x, t):
	gL = 25e-9
	Cm = 0.5e-9
	VL = -70e-3
	I = 2.5e-9

	dx = (-1*gL*(x - VL) + I) / Cm
	return dx

def main():
	dt = 0.0001
	duration = 2
	inits = [-7e-2]

	integrator = bp.fde.CaputoL1Schema(test_system, alpha = 1, num_memory = int(duration/dt), inits = inits)

	runner = bp.IntegratorRunner(integrator, monitors = list('x'), inits = inits, dt = dt)

	runner.run(duration)

	time = list()
	for i in range(int(duration/dt)):
		time.append(dt * i)


	f = open("data.txt", "w")
	now = datetime.datetime.now()
	f.write(str(now) + "\n")
	for i in range(int(duration/dt)):
		f.write(str(runner.mon.x[i]) + "\n")

	f.close()

	plt.figure(figsize=(10, 8))
	plt.plot(time, runner.mon.x)
	plt.show()
if __name__ == "__main__":
	main()
