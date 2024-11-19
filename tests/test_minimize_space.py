from numpy.random import Generator
from graphix.random_objects import rand_circuit, rand_gate

from graphix_or import minimize_space

def test_random_circuits(fx_rng: Generator) -> None:
   nqubits = 6
   depth = 6
   circuit = rand_circuit(nqubits, depth, fx_rng)
   pattern = circuit.transpile().pattern
   pattern.minimize_space()
   assert pattern.max_space() == 7
   pattern.perform_pauli_measurements()
   pattern.minimize_space()
   assert pattern.max_space() == 10
   minimize_space(pattern)
   assert pattern.max_space() == 7

def test_minimize_space_graph_maxspace_pauli(fx_rng: Generator) -> None:
   nqubits = 2
   depth = 4
   pairs = [(i, (i + 1) % nqubits) for i in range(nqubits)]
   circuit = rand_gate(nqubits, depth, pairs, fx_rng)
   pattern = circuit.transpile().pattern
   pattern.minimize_space()
   assert pattern.max_space() == 3
   pattern.perform_pauli_measurements()
   pattern.minimize_space()
   assert pattern.max_space() == 6
   minimize_space(pattern)
   assert pattern.max_space() == 5
