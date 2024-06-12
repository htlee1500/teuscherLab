import numpy as np


def main():
	layer1 = [[1, 0], [0, 0], [0, 0]]
	layer2 = [[2, 0], [0, 0], [0, 0]]
	layer3 = [[3,0], [0,0],[0,0]]
	layer4 = [[4,0], [0,0],[0,0]]


	brick = list()
	brick.append(layer1)
	brick.append(layer2)
	brick.append(layer3)
	brick.append(layer4)
	print(brick)

	brick = np.transpose(brick, (2, 1, 0))
	print(brick)


	vector = [1, 1, 1, 1]

	test = np.dot(brick, vector)
	print(test)

def getDims(matrix):
	print(str(len(matrix)) + ';' + str(len(matrix[0])) + ';' + str(len(matrix[0][0])))




if __name__ == '__main__':
	main()
