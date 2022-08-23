import numpy as np

'''
[github.com/ratdey]

Value estimator class
- estimates value of a state based on the generated beta vector
- uses F_BASE, STATEDIM and BETAFILEPATH parameters from the LSTD model.
- F_BASE, STATEDIM should match with the corresponding LSTD model for a given beta vector.
- state can be a python or numpy vector/matrix. 
'''


BETAFILEPATH = "./beta.txt"

F_BASE = 3
STATEDIM = 4

class Estimator:
	def __init__(self, file):
		self.stateDim = STATEDIM
		self.basis = F_BASE**self.stateDim
		# load beta
		self.beta = np.loadtxt(file)

		#generate Fourier matrix C
		self.FMat = np.zeros((self.basis, self.stateDim))
		self.FRow = np.zeros(self.stateDim)
		for x in range(self.basis):
			self.FMat[x] = self.FRow
			self.getNextFRow()

	# generates rows for Fourier matrix
	def getNextFRow(self):
		for i in range(self.stateDim):
			self.FRow[i] += 1
			if (self.FRow[i] <= F_BASE - 1):
				break
			self.FRow[i] = 0
		return

	def getPhiVec(self, state):
		return np.cos(np.dot(self.FMat, state) * np.pi)

	def estimateVal(self, state):
		state = np.array(state).flatten()
		phi = self.getPhiVec(state)
		return np.dot(self.beta, phi)


if __name__ == '__main__':
	estimator = Estimator(BETAFILEPATH)

	# calculate estimated value of the state
	# estimator.estimateVal(stateFeatures)

	# demo
	state1 = [[0, 1], [0.32, 0.0]]
	val = estimator.estimateVal(state1)
	print(val)
	state2 = np.array([[22, 7], [0.82, 0.1]])
	val = estimator.estimateVal(state2)
	print(val)
	state3 = [1, 7, 0.31, 0.1]
	val = estimator.estimateVal(state3)
	print(val)






