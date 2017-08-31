#!/usr/bin/env python

from helper_foo import *

"""Runs a genetic/evolutionary algorithm that finds a solution to the graph colouring problem"""

__author__ = "Jan Krepl"
__copyright__ = "Copyright 2017, Jan Krepl"
__license__ = "MIT"
__version__ = "1.0"
__email__ = "jankrepl@yahoo.com"
__status__ = "Production"

# PARAMETERS
population_size = 500  # Stays constant throughout evolution
n_nodes = 25  # Number of countries = number of nodes in a graph
number_of_edges, al = generate_random_graph(n_nodes, 0.6)  # The graph architecture is generated randomly
n_generations = 100
genetic_op = 'mutation'  # 'SPC' - single point crossover or 'mutation'
percentage_of_parents_to_keep = 0.2  # In each generation update a certain percentage of fittest parents is kept

# NUMBER OF EDGES
print("Number of edges: " + str(number_of_edges))

# MAIN ALGORITHM
input_population = generate_random_initial_population(population_size, n_nodes, al)
results_fitness, results_fittest = evolution(input_population, n_generations, population_size,
                                             percentage_to_keep=percentage_of_parents_to_keep,
                                             genetic_op=genetic_op)

# VISUALIZE
visualize_results(results_fitness, results_fittest)
plt.show()
