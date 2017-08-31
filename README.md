# Graph Colouring

Colouring of an undirected graph with **3 colors** - red, green and blue. The goal is to minimize the number of edges that
connect nodes with the same color. We apply a **genetic/evolutionary algorithm**.

## Detailed Description
### Problem
The above described problem has a perfect solution when we are allowed to use **4 colors** (see [Four Color Theorem](https://en.wikipedia.org/wiki/Four_color_theorem)).
However, in the 3 color case the perfect solution does not exist in general. The problem can be also thought of as colouring countries
on a map in a way that countries that are sharing a border have different colors.

### Solution Approach
We employ a **genetic/evolutionary algorithm** to find possible solutions and visualize results. To each possible graph colouring we
assign a **fitness measure** = number of edges that connect nodes with different colours. Based on this measure we 
determine breeding probabilities that are then use in parent selection - **Fitness Proportional Selection**. On top of that, 
we guarantee that a certain percentage of fittest parents is always copied into a new generation (**Elitism**). There are
two available genetic operators for breeding - **Mutation** and **Single point crossover**.


## Code Preliminaries



### Running


```
python Main.py
```

### Dependencies
*  matplotlib
*  networkx
*  numpy

## Parameters and methods
We will refer to one possible colouring as a **person**

* population_size 
  * Number of people in our population - it is kept constant in each generation
* n_nodes
  * Number of nodes (countries) on a map
  * We generate a random graph with the help of networkx package
* n_generations
  * Number of generations that our population is going to evolve for
* genetic_op
  * possible values: 'mutation' or 'SPC'
    * *Mutation* picks a random node within the graph and changes its the color randomly
    * *Single Point Crossover* picks a random node **n** and defines a child graph by putting the first **n** nodes equal
    to Parent_1 nodes the following nodes equal to Parent_2
* percentage_of_parents_to_keep
  * Percentage of fittest parents to automatically copy into a new generation
  
## Example
```
n_nodes = 25
n_edges_total = 207 (implied by random graph)
population_size = 1000
n_generations = 60
genetic_op = 'mutation'
```
**1st generation fittest colouring**
![gener_1](https://user-images.githubusercontent.com/18519371/29922174-c8a03cfc-8e54-11e7-8795-d758d5a28c8e.png)

**60th generation fittest colouring**
![gener_60](https://user-images.githubusercontent.com/18519371/29922216-ed5b4a78-8e54-11e7-9363-76ba49828e97.png)

## Possible extensions
* Hyperparameter optimizar - Grid Search
* Implementation of Island Model


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

Idea from the book: Machine Learning: An Algorithmic Perspective Second Edition - Stephen Marsland
