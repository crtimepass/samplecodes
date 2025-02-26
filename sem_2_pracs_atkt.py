"""
# Prac 1 AND OR NOT Gate
"""

class McCullohPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights = weights
        self.threshold = threshold
    def activate(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return 1 if weighted_sum >= self.threshold else 0

def and_gate(input_1, input_2):
    weights = [1, 1]
    threshold = 2
    neuron = McCullohPittsNeuron(weights, threshold)
    return neuron.activate([input_1, input_2])

def or_gate(input_1, input_2):
    weights = [1, 1]
    threshold = 1
    neuron = McCullohPittsNeuron(weights, threshold)
    return neuron.activate([input_1, input_2])

def not_gate(input_1):
    weights = [-1]
    threshold = 0
    neuron = McCullohPittsNeuron(weights, threshold)
    return neuron.activate([input_1])

print("AND(0,0): ",and_gate(0, 0))
print("AND(0,1): ",and_gate(0, 1))
print("AND(1,0): ",and_gate(1, 0))
print("AND(1,1): ",and_gate(1, 1))
print("OR(0,0): ",or_gate(0, 0))
print("OR(0,1): ",or_gate(0, 1))
print("OR(1,0): ",or_gate(1, 0))
print("NOT(0): ",not_gate(0))
print("NOT(1): ",not_gate(1))

"""# Prac 2 Hebb's Rule"""

class HebianNeuron:
    def __init__(self, input_size):
        self.weights = [0.0] * input_size

    def activate(self, inputs):
        weighted_sum = sum(w * i for w, i in zip(self.weights, inputs))
        return 1 if weighted_sum > 1 else 0

    def train(self, inputs, target):
        for i in range(len(self.weights)):
            self.weights[i] += inputs[i] * target

training_data = [
 ([0,0], 0),
 ([0,1], 0),
 ([1,0], 0),
 ([1,1], 1)
]

neuron = HebianNeuron(2)

for inputs, target in training_data:
    neuron.train(inputs, target)

test_data = [
 [0,0],
 [0,1],
 [1,0],
 [1,1]
]


print("Weights after training:", neuron.weights)
for inputs in test_data:
    print(f"Input: {inputs}, Output: {neuron.activate(inputs)}")

"""# Prac 3 Kohonen Self Organizing Map"""

import numpy as np
import matplotlib.pyplot as plt

class KohonenSOM:
    def __init__(self, input_dim, grid_size, learning_rate = 0.1, radius = None, epochs = 1000):
        self_input_dim = input_dim
        self.grid_size = grid_size
        self.weights = np.random.rand(grid_size[0], grid_size[1], input_dim)
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(grid_size) / 2
        self.epochs = epochs

    def train(self, data):
        time_constant = self.epochs / np.log(self.radius)
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            for sample in data:
                bmu_index = self._find_bmu(sample)
                self._update_weights(sample, bmu_index, epoch, time_constant)

    def _find_bmu(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), (self.grid_size[0], self.grid_size[1]))

    def _update_weights(self, sample, bmu_index, epoch, time_constant):
        learning_rate = self.learning_rate * np.exp(-epoch / self.epochs)
        radius = self.radius * np.exp(-epoch / time_constant)
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                neuron_pos = np.array([i,j])
                bmu_pos = np.array(bmu_index)
                distance = np.linalg.norm(neuron_pos - bmu_pos)
                if distance < radius:
                    influence = np.exp(-distance**2 / (2 * (radius**2)))
                    self.weights[i,j] += influence * learning_rate * (sample - self.weights[i,j])

    def map_vects(self, data):
        mapped = np.array([self._find_bmu(sample) for sample in data])
        return mapped

data = np.random.rand(100, 3)
som = KohonenSOM(input_dim=3, grid_size=(10,10), learning_rate=0.1, epochs=1000)
som.train(data)
mapped_data = som.map_vects(data)
plt.scatter(mapped_data[:,0], mapped_data[:,1])
plt.title("Data mapped to SOM grid")
plt.show()

"""
# Prac 4 Hamming Network"""

import numpy as np

def hamming_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def feedforward(input_vec, exemplar_vecs):
    N = len(input_vec)
    distances = np.array([hamming_distance(input_vec, vec) for vec in exemplar_vecs])
    y = N - distances
    return y

def recurrent_phase(y, aplha=0.5, iterations=10):
    y = np.array(y, dtype=float)
    for _ in range(iterations):
        y_new = y - aplha * (np.sum(y)-y)
        if np.allclose(y, y_new, atol = 1e-6):
            break
        y = y_new
        return y

exemplar_vecs = np.array([
 [1, 0, 1, 0],
 [0, 1, 0, 1]
])

input_vec = np.array([1, 1, 0, 0])
y_feedforward = feedforward(input_vec, exemplar_vecs)
print("Feedforward outputs:" , y_feedforward)
y_recurrent = recurrent_phase(y_feedforward)
print("Recurrent outputs:", y_recurrent)

"""# Prac 5 BAM Network"""

import numpy as np

class BAM:
	def __init__(self):
		self.weights = None

	def train(self, patterns_A, patterns_B):
		num_features_A = patterns_A.shape[1]
		num_features_B = patterns_B.shape[1]
		self.weights = np.zeros((num_features_A, num_features_B))
		for a, b in zip(patterns_A, patterns_B):
			self.weights += np.outer(a, b)

	def recall_A(self, pattern_B):
		result = np.dot(pattern_B, self.weights.T)
		return np.sign(result)

	def recall_B(self, pattern_A):
		result = np.dot(pattern_A, self.weights)
		return np.sign(result)

patterns_A = np.array([[1, 1, -1], [-1, 1, 1], [-1, -1, -1]])
patterns_B = np.array([[1, -1], [-1, 1], [1, 1]])
bam = BAM()
bam.train(patterns_A, patterns_B)
test_pattern_B = np.array([1, -1])
recalled_pattern_A = bam.recall_A(test_pattern_B)
print("Recalled Pattern A for test pattern B", test_pattern_B, "is:", recalled_pattern_A)
test_pattern_A = np.array([1, 1, -1])
recalled_pattern_B = bam.recall_B(test_pattern_A)
print("Recalled Pattern B for test pattern A", test_pattern_A, "is:", recalled_pattern_B)

"""# Prac 6 MaxNet"""

def winner_takes_all(y_out):
	return [i for i, val in enumerate(y_out) if val > 0]

def maxnet():
	m = int(input("Enter the number of nodes: "))
	delta = float(input("Enter delta value: "))
	y_in = [0.0] * m
	for i in range(m):
		y_in[i] = float(input(f"Enter the initial value of node {i+1}: \t "))

	epoch = 0
	while True:
		epoch += 1
		y_out = []
		for i in range(m):
			if (y_in[i]) >= 0:
				y_out.append(y_in[i])
			else:
				y_out.append(0)
		if len(winner_takes_all(y_out)) == 1:
			print("Winner is unit: ", winner_takes_all(y_out)[0] + 1)
			break
		for i in range(m):
			y_in[i] = y_out[i] - (sum(y_out) - y_out[i]) * delta
		if epoch == 100:
			print("Maximum number of epoch limit reached.")
			break
		print(f"Epoch {epoch}: y_out = {y_out}")

maxnet()

"""# Prac 7 De-Morgan's Law"""

def de_morgans_law_1(A, B):
    not_A_or_B = not (A or B)
    not_A_and_not_B = (not A) and (not B)
    return not_A_or_B, not_A_and_not_B

def de_morgans_law_2(A, B):
    not_A_and_B = not (A and B)
    not_A_or_not_B = (not A) or (not B)
    return not_A_and_B, not_A_or_not_B

A = bool(int(input("Enter A (0 or 1): ")))
B = bool(int(input("Enter B (0 or 1): ")))

result_1 = de_morgans_law_1(A, B)
print("De Morgan's Law 1: ~(A v B) = ~A ∧ ~B\n")
print(f"~({A} v {B}) = {result_1[0]}")
print(f"~{A} ∧ ~{B} = {result_1[1]}")
print(f"Law holds: {result_1[0] == result_1[1]}\n")

result_2 = de_morgans_law_2(A, B)
print("De Morgan's Law 2: ~(A ∧ B) = ~A ∨ ~B")
print(f"~({A} ∧ {B}) = {result_2[0]}")
print(f"~{A} ∨ ~{B} = {result_2[1]}")
print(f"Law holds: {result_2[0] == result_2[1]}")

"""# Prac 8 Fuzzy Union, Intersection, Complement and Difference"""

def fuzzy_union(set1, set2):
	"""Union of two fuzzy sets."""
	return {key: max(set1.get(key, 0), set2.get(key, 0)) for key in set(set1) | set(set2)}

def fuzzy_intersection(set1, set2):
	"""Intersection of two fuzzy sets."""
	return {key: min(set1.get(key, 0), set2.get(key, 0)) for key in set(set1) | set(set2)}

def fuzzy_complement(set1):
	"""Complement of a fuzzy set."""
	return {key: 1 - value for key, value in set1.items()}

def fuzzy_difference(set1, set2):
	"""Difference between two fuzzy sets."""
	return {key: min(set1.get(key, 0), 1 - set2.get(key, 0)) for key in set(set1) | set(set2)}

if __name__ == "__main__":
	fuzzy_set_A = {'a': 0.7, 'b': 0.5, 'c': 0.2}
	fuzzy_set_B = {'b': 0.6, 'c': 0.4, 'd': 0.9}
	print("Fuzzy Set A:", fuzzy_set_A)
	print("Fuzzy Set B:", fuzzy_set_B)
	union_result = fuzzy_union(fuzzy_set_A, fuzzy_set_B)
	print("Union of A and B:", union_result)
	intersection_result = fuzzy_intersection(fuzzy_set_A, fuzzy_set_B)
	print("Intersection of A and B:", intersection_result)
	complement_result_A = fuzzy_complement(fuzzy_set_A)
	print("Complement of A:", complement_result_A)
	difference_result = fuzzy_difference(fuzzy_set_A, fuzzy_set_B)
	print("Difference of A and B:", difference_result)

"""# Prac 9 Fuzzy Cartesian Product"""

import matplotlib.pyplot as plt
import numpy as np

def cartesian_product_fuzzy_relation(A, B):
	"""
	Create a fuzzy relation by Cartesian product of fuzzy sets A
	and B.
	The membership value of the pair (x, y) is min(A(x), B(y)).
	"""
	relation = {}
	for x in A:
		for y in B:
			relation[(x, y)] = min(A[x], B[y])
	return relation

def max_min_composition(R, S):
	"""
	Perform max-min composition on fuzzy relations R and S.
	"""
	T = {}
	x_elements = set(x for x, y in R)
	y_elements = set(y for x, y in R)
	z_elements = set(z for y, z in S)
	for x in x_elements:
		for z in z_elements:
			min_values = []
			for y in y_elements:
				if (x, y) in R and (y, z) in S:
					min_values.append(min(R[(x, y)], S[(y, z)]))
			if min_values:
				T[(x, z)] = max(min_values)
	return T

def plot_fuzzy_set(fuzzy_set, title):
	"""
	Plot a fuzzy set.
	"""
	elements = list(fuzzy_set.keys())
	membership_values = list(fuzzy_set.values())
	plt.figure()
	plt.bar(elements, membership_values, color='skyblue')
	plt.ylim(0, 1)
	plt.xlabel('Elements')
	plt.ylabel('Membership Values')
	plt.title(title)
	plt.grid(axis='y', linestyle='--')
	plt.show()

def plot_fuzzy_relation(relation, x_label, y_label, title):
	"""
	Plot a fuzzy relation as a heatmap.
	"""
	x_elements = list(set(x for x, y in relation))
	y_elements = list(set(y for x, y in relation))
	X, Y = np.meshgrid(range(len(x_elements)), range(len(y_elements)))
	Z = np.zeros(X.shape)
	for i, x in enumerate(x_elements):
		for j, y in enumerate(y_elements):
			Z[j, i] = relation.get((x, y), 0)
	plt.figure()
	plt.imshow(Z, aspect='auto', cmap='viridis', origin='lower')
	plt.colorbar(label='Membership Values')
	plt.xticks(range(len(x_elements)), x_elements)
	plt.yticks(range(len(y_elements)), y_elements)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.show()

def plot_3d_relation(relation, x_label, y_label, z_label, title):
	"""
	Plot a fuzzy relation as a 3D surface plot.
	"""
	x_elements = list(set(x for x, y in relation))
	y_elements = list(set(y for x, y in relation))
	X, Y = np.meshgrid(range(len(x_elements)), range(len(y_elements)))
	Z = np.zeros(X.shape)
	for i, x in enumerate(x_elements):
		for j, y in enumerate(y_elements):
			Z[j, i] = relation.get((x, y), 0)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(X, Y, Z, cmap='viridis')
	ax.set_xticks(range(len(x_elements)))
	ax.set_xticklabels(x_elements)
	ax.set_yticks(range(len(y_elements)))
	ax.set_yticklabels(y_elements)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.set_zlabel(z_label)
	ax.set_title(title)
	plt.show()

A = {'x1': 0.7, 'x2': 0.4, 'x3': 0.9}
B = {'y1': 0.6, 'y2': 0.8, 'y3': 0.5}
C = {'z1': 0.5, 'z2': 0.9, 'z3': 0.3}
R = cartesian_product_fuzzy_relation(A, B)
S = cartesian_product_fuzzy_relation(B, C)
T = max_min_composition(R, S)
plot_fuzzy_set(A, "Fuzzy Set A")
plot_fuzzy_set(B, "Fuzzy Set B")
plot_fuzzy_set(C, "Fuzzy Set C")
plot_fuzzy_relation(R, 'Elements of A', 'Elements of B', "Fuzzy Relation R (A × B)")
plot_fuzzy_relation(S, 'Elements of B', 'Elements of C', "Fuzzy Relation S (B × C)")
plot_3d_relation(T, 'Elements of A', 'Elements of C', 'Membership Values', "Max-Min Composition (R o S)")

"""# Prac 10 Max-min Composition Fuzzy"""

def cartesian_product_fuzzy_relation(A, B):
	"""
	Create a fuzzy relation by Cartesian product of fuzzy sets A
	and B.
	The membership value of the pair (x, y) is min(A(x), B(y)).
	"""
	relation = {}
	for x in A:
		for y in B:
			relation[(x, y)] = min(A[x], B[y])
	return relation

def max_min_composition(R, S):
	"""
	Perform max-min composition on fuzzy relations R and S.
	"""
	T = {}
	x_elements = set(x for x, y in R)
	y_elements = set(y for x, y in R)
	z_elements = set(z for y, z in S)
	for x in x_elements:
		for z in z_elements:
			min_values = []
			for y in y_elements:
				if (x, y) in R and (y, z) in S:
					min_values.append(min(R[(x, y)], S[(y, z)]))
			if min_values:
				T[(x, z)] = max(min_values)
	return T

# Predefined fuzzy sets
A = {'x1': 0.7, 'x2': 0.4, 'x3': 0.9}
B = {'y1': 0.6, 'y2': 0.8, 'y3': 0.5}
C = {'z1': 0.5, 'z2': 0.9, 'z3': 0.3}

# Compute fuzzy relations
R = cartesian_product_fuzzy_relation(A, B)
S = cartesian_product_fuzzy_relation(B, C)

# Perform max-min composition
T = max_min_composition(R, S)

# Print the results
print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)
print("Fuzzy Set C:", C)
print("\nFuzzy Relation R (A × B):", R)
print("Fuzzy Relation S (B × C):", S)
print("\nMax-Min Composition (R o S):")
for (x, z), value in T.items():
    print(f"({x}, {z}): {value}")

