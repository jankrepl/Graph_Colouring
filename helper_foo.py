""" Helper Functions and Objects
"""

import matplotlib.pyplot as plt  # Drawing graphs
import networkx as nx  # Generating of graphs
import numpy as np


# Define the main data structure
class World_Map:
    def __init__(self, colors, adjacency_list):
        """
        Initialization of World_Map object

        :param colors: String of 'r', 'g' or 'b' - e.g. 'rbbgrgbg'
        :type colors: str
        :param adjacency_list: adjacency list - list of neighbours for each node
        :type adjacency_list: list of lists
        """
        self.colors = colors  # string .. i-th element represents color - 3 possible colours 'r','g' and 'b'
        self.adjacency_list = adjacency_list  # list of lists
        self.n_nodes = len(self.colors)
        self.fitness, self.graph_nx = self.__convert_to_nxgraph(self.colors, self.adjacency_list)

    # The networks package offers amazing visualization features
    def __convert_to_nxgraph(self, colors, adjacency_list):
        """
        Generates a networkx graph object and in the meantime calculates fitness

        :param colors: String of 'r', 'g' or 'b' - e.g. 'rbbgrgbg'
        :type colors: str
        :param adjacency_list: adjacency list - list of neighbours for each node
        :type adjacency_list: list of lists
        :return: (fitness, networkx graph object)
        :rtype: (int, netowrkx Graph)
        """
        G = nx.Graph()
        counter = 0  # the number of edges connecting same colors
        number_of_edges_twice = 0

        for index, node_color in enumerate(colors):
            G.add_node(index, color=node_color)  # Index the label
            for neighbour in adjacency_list[index]:
                # input edge
                G.add_edge(index, neighbour, illegal=False)  # illegal bool denotes whether nodes of the same color
                number_of_edges_twice += 1
                if node_color == colors[neighbour]:
                    G[index][neighbour]['illegal'] = True
                    counter += 1

        return number_of_edges_twice / 2 - counter / 2, G

    def print_me(self, figure_number=-1, figure_title=''):
        """
        Prints a graph representing the object

        :param figure_number: number of the figure
        :type figure_number: int
        :param figure_title: name of the figure
        :type figure_title: str

        """
        color_mapping = {True: 'r', False: 'g'}

        node_list = self.graph_nx.nodes(data=True)
        edge_list = self.graph_nx.edges(data=True)

        colors_nodes = [element[1]['color'] for element in node_list]
        colors_edges = [color_mapping[element[2]['illegal']] for element in edge_list]
        plt.figure(figure_number)
        plt.title(figure_title)
        nx.draw_networkx(self.graph_nx, with_labels=True, node_color=colors_nodes, edge_color=colors_edges)
        plt.draw()


# Generate random graph - in our case we care about adjacency list
def generate_random_graph(number_of_nodes, probability_of_edge):
    """

    Generates a random graph that has on average probability_of_edge * (number_of_nodes choose 2) edges

    :param number_of_nodes: number of graph nodes
    :type number_of_nodes: int
    :param probability_of_edge: Probability of an edge given any two different nodes
    :type probability_of_edge: float
    :return: (number of edges, adjacency list)
    :rtype: (int, list of lists)
    """

    G = nx.fast_gnp_random_graph(number_of_nodes, probability_of_edge, seed=None, directed=False)
    edges = []
    for i in range(number_of_nodes):
        temp1 = G.adj[i]
        edges.append(list(G.adj[i].keys()))
    return G.number_of_edges(), edges


# Creates list of pairs from the input population
def parent_selection(input_population, number_of_pairs, method='FPS'):
    """
    Forms pairs from the input population (with replacement)

    :param input_population:
    :type input_population: list of World_Map
    :param number_of_pairs: number of desired output pairs -> output population = 2 * number_of_pairs
    :type number_of_pairs: int
    :param method: selection method
    :type method: str
    :return: paired up parents from the input population
    :rtype: list of pairs of World_Map
    """

    # Useful
    input_n = len(input_population)

    if method == 'FPS':  # Fitness proportional selection
        # our fitness is non-negative so we can apply a simple formula  fitness_m/sum(fitness_i)
        fitness_sum = sum([person.fitness for person in input_population])
        probabilities = np.array([person.fitness / fitness_sum for person in input_population])

        I_x = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)
        I_y = np.random.choice(np.arange(0, input_n), number_of_pairs, p=probabilities)

        return [(input_population[I_x[i]], input_population[I_y[i]]) for i in range(number_of_pairs)]


# Define a genetic operator
def genetic_operator(pair_of_parents, method='SPC'):
    """
    For a given pair of parents we output a pair of children based one a given genetic operator

    :param pair_of_parents: pair of parents
    :type pair_of_parents: pair of World_Map
    :param method: genetic operator method, 'SPC' - single point crossover
    :type method: str
    :return: pair of children
    :rtype: pair of World_Map
    """

    n_nodes = pair_of_parents[0].n_nodes
    al = pair_of_parents[0].adjacency_list

    if method == 'mutation':
        # this method does not need a pair of parents, it inputs only one person
        # Idea:

        node1 = np.random.randint(0, n_nodes)
        node2 = np.random.randint(0, n_nodes)

        mapper = {'r': ['b', 'g'], 'b': ['r', 'g'], 'g': ['r', 'b']}

        child_one_colors = pair_of_parents[0].colors
        child_two_colors = pair_of_parents[1].colors

        child_one_colors = child_one_colors[:node1] + np.random.choice(mapper[child_one_colors[node1]],
                                                                       1)[0] + child_one_colors[node1 + 1:]
        child_two_colors = child_two_colors[:node2] + np.random.choice(mapper[child_two_colors[node2]],
                                                                       1)[0] + child_two_colors[node2 + 1:]

        return World_Map(child_one_colors, al), World_Map(child_two_colors, al)

    if method == 'SPC':  # Single point crossover
        # Step 1) Select a random point
        # Step 2) All colours to the left will be from parent 1, all parent to the right are from parent 2
        point = np.random.randint(0, n_nodes)

        parent_1_colors = pair_of_parents[0].colors
        parent_2_colors = pair_of_parents[0].colors

        child_one_colors = parent_1_colors[:point] + parent_2_colors[point:]
        child_two_colors = parent_2_colors[:point] + parent_1_colors[point:]

        return (World_Map(child_one_colors, al), World_Map(child_two_colors, al))


# Population update
def population_update(input_population, output_population_size, generation_change_method='Elitism',
                      percentage_to_keep=0.1, genetic_op='SPC'):
    """
    Population update step

    :param input_population: input population
    :type input_population: list of World_Map
    :param output_population_size: size of output population
    :type output_population_size: int
    :param generation_change_method: method that determines how some parents could move to new generation
    :type generation_change_method: str
    :param percentage_to_keep: percentage of fittest parents to transfer to new generation, in (0,1)
    :type percentage_to_keep: float
    :param genetic_op: genetic operator used for breeding
    :type genetic_op: str
    :return: output population
    :rtype: list of World_Map
    """
    input_population_size = len(input_population)
    output_population = []

    if generation_change_method == 'Elitism':
        #  # We keep the best x percent of the input population
        input_population.sort(key=lambda x: x.fitness, reverse=True)
        output_population += input_population[:int(input_population_size * percentage_to_keep)]

        list_of_parent_pairs = parent_selection(input_population, input_population_size // 2)

        pair_index = 0
        while len(output_population) < output_population_size:
            child_1, child_2 = genetic_operator(list_of_parent_pairs[pair_index], method=genetic_op)
            output_population.append(child_1)
            output_population.append(child_2)
            pair_index += 1

    return output_population


# Generate random initial population
def generate_random_initial_population(population_size, n_nodes, al):
    """
    Randomly create an initial population

    :param population_size: population size
    :type population_size: int
    :param n_nodes: number of nodes
    :type n_nodes: int
    :param al: adjacency list
    :type al: list of lists
    :return: random population
    :rtype: list of World_Map
    """
    input_population = []

    # Generate random initial population
    for _ in range(population_size):
        color_list = np.random.choice(['r', 'b', 'g'], n_nodes, replace=True)
        color_string = "".join(color_list)
        input_population.append(World_Map(color_string, al))
    print('A random population of ' + str(population_size) + ' people was created')

    return input_population


# Find fittest
def find_fittest(input_population):
    """
    Given a population, find the fittest person

    :param input_population: input population
    :type input_population: list of World_Map
    :return: (list of fitness values for entire population, index of the fittest person, the fittest person)
    :rtype: (list, int, World_Map)
    """
    fitness_list = [person.fitness for person in input_population]
    ix = np.argmax(fitness_list)
    return fitness_list, ix, input_population[ix]


# Roll the evolution
def evolution(input_population, n_generations, population_size, percentage_to_keep=0.1, genetic_op='SPC'):
    """
    Iterative update of generations

    :param input_population: input population
    :type input_population: list of World_Map
    :param n_generations: number of generations to simulate
    :type n_generations: int
    :param population_size: desired population size - it stays constant throughout evolution
    :type population_size: int
    :param percentage_to_keep:  percentage of fittest parents to transfer into a new generation, in (0,1)
    :type percentage_to_keep: float
    :param genetic_op: genetic operation for breeding
    :type genetic_op: str
    :return: (for each generation list of fitness of each person, for each generation the fittest person)
    :rtype: (list of lists, list of World_Map)
    """
    # We will find the histogram for each generation and the fittest person
    results_fitness = []
    results_fittest = []

    for i in range(n_generations):
        print('Your population is in the ' + str(i + 1) + '-th generation')
        # Save results
        fitness_list, ix, fittest_coloring = find_fittest(input_population)
        results_fitness.append(fitness_list)
        results_fittest.append(fittest_coloring)
        # Print highest fitness
        print('The fittest person is: ' + str(max(fitness_list)))

        # Update
        output_population = population_update(input_population, population_size,
                                              percentage_to_keep=percentage_to_keep, genetic_op=genetic_op)
        input_population = output_population

    return results_fitness, results_fittest


# Visualize results
def visualize_results(results_fitness, results_fittest, number_of_generations_to_visualize=6):
    """
    Visualization of the evolution

    :param results_fitness: for each generation list of fitness of each person
    :type results_fitness: list of lists
    :param results_fittest: for each generation the fittest person
    :type results_fittest: list of World_Map
    :param number_of_generations_to_visualize: number of separate generations to visalize - 0, last and random generations
    :type number_of_generations_to_visualize: int
    :return: Plots the fittest person (graph) and histogram for each generation
    """
    # Important
    total_generations = len(results_fitness)

    # Pick generations to visualize
    I = list(
        np.random.choice(list(range(1, total_generations - 1)), number_of_generations_to_visualize - 2, replace=False))
    I += [0, total_generations - 1]
    I.sort()
    print("Visualized generations: ",end='')
    print(I)

    # Main
    for i, order in enumerate(I):
        # print fittest
        results_fittest[order].print_me(figure_number=i,
                                        figure_title='generation: ' + str(order + 1) + ', fitness: ' + str(
                                            results_fittest[order].fitness))
        # Print histogram
        plt.figure(-i - 1)  # means nothing
        plt.hist(results_fitness[order])
        plt.title('Generation_number: ' + str(order + 1))
        plt.draw()
