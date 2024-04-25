import math


def mittag_leffler(beta=0.2, tau=0.02, t=1e-4):

	E = 0
	z = 1 * t**beta / tau

	p0 = 0
	p1 = -1 * (math.gamma(1-beta)**2 * math.gamma(1+beta) / math.gamma(1-2*beta)) + math.gamma(1+beta)
	p1 = p1 / (math.gamma(1+beta) * math.gamma(1-beta) - 1)

	q0 = -1 * (math.gamma(1-beta) * math.gamma(1+beta) / math.gamma(1-2*beta)) + math.gamma(1+beta)/math.gamma(1-beta)
	q0 = q0 / (math.gamma(1+beta) * math.gamma(1-beta) - 1)

	q1 = math.gamma(1+beta) - math.gamma(1-beta)/math.gamma(1-2*beta)
	q1 = q1 / (math.gamma(1+beta) * math.gamma(1-beta) - 1)

	E = 1 / math.gamma(1-beta) * (p1 + z) / (q0 + q1*z + z**2)


	return E
