import math

def generate_gamma_single(B=0.2, n=100):

	em = 0.577216

	gamma_values = list()

	for k in range(n):
		z = B*k + 1

		gamma = z * math.e**(em*z)

		for i in range(1, 51):
			gamma = gamma * (1+z/n) * math.e**(-1 * z / n)

		gamma = gamma**-1

		gamma_values.append(gamma)

	return gamma_values

def generate_gamma_double(B=0.2, n=100):

	em = 0.577216

	gamma_values = list()

	for k in range(n):

		z = B*k + B

		gamma = z * math.e**(em*z)

		for i in range(1, 51):
			gamma = gamma * (1+z/n) * math.e**(-1 * z / n)

		gamma = gamma**-1

		gamma_values.append(gamma)

	return gamma_values

def compute_ML_single(B=0.2, tau, t, n=100, gamma_single):

	E = 0
	for i in range(n):
		E += (-1 * t**B / tau)**i / gamma_single[i]

	return E

def compute_ML_double(B=0.2, tau, t, n=100, gamma_double):

	E = 0
	for i in range(n):
		E += (-1 * t**B / tau)**i / gamma_double[i]

	return E
