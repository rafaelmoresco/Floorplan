import math
import random
from dataStructures import Circuit, BStarTree, BStarTreeNode

class Annealer:
    def __init__(self, circuit: Circuit):
        self.circuit = circuit

    def anneal(self, initial_temp: float = 1000.0, cooling_rate: float = 0.95, iterations_per_temp: int = 100, min_temp: float = 1.0):
        """
        Perform simulated annealing to optimize floorplanning.
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Rate at which temperature decreases
            iterations_per_temp: Number of iterations at each temperature
            min_temp: Minimum temperature to stop annealing
        """
        # Initialize with random B* tree
        self.circuit.circuit_to_bstar_random(seed=random.randint(1, 1000))
        self.circuit.bstar_to_circuit()
        
        current_width, current_height, current_area = self.circuit.get_floorplan_area()
        best_area = current_area
        best_tree = self.circuit.bstar_tree.copy()
        
        temperature = initial_temp
        
        while temperature > min_temp:
            for _ in range(iterations_per_temp):
                # Generate neighbor by randomly choosing an operation
                neighbor_tree = self.circuit.bstar_tree.copy()
                nodes = neighbor_tree.collect_nodes()
                if len(nodes) < 2:
                    continue
                op = random.choice(["swap"])
                #op = "swap"
                if op == "swap":
                    node1, node2 = random.sample(nodes, 2)
                    node1.data, node2.data = node2.data, node1.data
                elif op == "rotate":
                    node = random.choice(nodes)
                    neighbor_tree.rotate_macro(node.data)
                elif op == "move":
                    node = random.choice(nodes)
                    neighbor_tree.move_macro_to_random(node.data)
                # Apply neighbor tree to circuit and calculate area
                self.circuit.bstar_tree = neighbor_tree
                self.circuit.bstar_to_circuit()
                new_width, new_height, new_area = self.circuit.get_floorplan_area()
                overlap = self.circuit.check_overlaps()
                # Calculate acceptance probability
                delta_e = new_area - current_area
                if (delta_e < 0 or random.random() < math.exp(-delta_e / temperature)) and not overlap:
                    # Accept the move
                    current_area = new_area
                    if new_area < best_area:
                        best_area = new_area
                        best_tree = neighbor_tree.copy()
                else:
                    # Reject the move, revert to previous state
                    self.circuit.bstar_tree = self.circuit.bstar_tree.copy()
            temperature *= cooling_rate
        # Apply the best solution found
        self.circuit.bstar_tree = best_tree
        self.circuit.bstar_to_circuit()
        return best_area
