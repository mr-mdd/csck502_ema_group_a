#!/usr/bin/env python3
# Python Interpreter


# Import Modules
# For generating random values for candidate grids
import random
# For duplicating INITIAL_GRID without altering it
import copy


# Configuration Constants
# Defines four unique letters used to fill 4x4 grid
LETTERS = ['W', 'D', 'R', 'O']
# Sets dimension of grid
GRID_SIZE = 4
# Specifies dimension of each subgrid within main grid
SUBGRID_SIZE = 2
# Determines number of candidate solutions in each generation of genetic algorithm
POPULATION_SIZE = 100
# Sets cap on number of iterations genetic algorithm will perform
GENERATIONS = 1000
# Indicates probability that a given candidate solution will undergo mutation
MUTATION_RATE = 0.1
# Initial grid as template for generating candidate solutions
INITIAL_GRID = [
    ['W', None, None, 'O'],
    [None, 'R', None, None],
    [None, None, 'D', None],
    ['D', None, None, 'R']
]


# Random Grid Generator
# Takes initial grid and returns a filled grid by randomly assigning letters to empty cells
def initialize_grid(initial_grid):
    # Ensures function works on duplicate of original grid
    grid = copy.deepcopy(initial_grid)

    # Iterates over each cell in grid by row (i) and column (j)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Checks if cell is empty and assigns random letter
            if grid[i][j] is None:
                grid[i][j] = random.choice(LETTERS)
    return grid


# Population Generator
# Takes random grid generator and returns a starting population of candidate solutions
def create_initial_population(initial_grid, population_size):
    # List to hold all candidate grids
    population = []

    # Loop iterating population_size times to generate candidates in population list
    for _ in range(population_size):
        candidate = initialize_grid(initial_grid)
        population.append(candidate)
    return population


# Fitness Score Calculator
# Takes candidate grids and returns a score based on number of conflicts, where 0 is best
def calculate_fitness(grid):
    # Variable to hold sum of conflicts
    conflicts = 0

    # Finds difference between total and unique number of elements for each row
    for row in grid:
        conflicts += len(row) - len(set(row))

    # Finds difference between total and unique number of elements for each column
    for j in range(GRID_SIZE):
        col = [grid[i][j] for i in range(GRID_SIZE)]
        conflicts += len(col) - len(set(col))

    # Partitions the grid into four 2x2 subgrids
    for sub_i in range(0, GRID_SIZE, SUBGRID_SIZE):
        for sub_j in range(0, GRID_SIZE, SUBGRID_SIZE):
            # Variable to hold 2x2 subgrid elements
            block = []
            # Loop iterating to collect subgrid elements in block list
            for i in range(sub_i, sub_i + SUBGRID_SIZE):
                for j in range(sub_j, sub_j + SUBGRID_SIZE):
                    block.append(grid[i][j])
            # Finds difference between total and unique number of elements in subgrids
            conflicts += len(block) - len(set(block))
    return conflicts


# Tournament Selection Mechanism
# Takes population list and returns two candidate grids (parents) for genetic operations
def selection(population):
    # Establishes number of candidates for tournament
    tournament_size = 5
    # Randomly selects unique candidates from population list
    tournament = random.sample(population, tournament_size)
    # Sorts candidates in ascending order based on their fitness scores
    tournament.sort(key=calculate_fitness)
    # Returns top 2 candidates
    return tournament[0], tournament[1]


# Crossover Child Grid Generator
# Takes two candidate grids (parents) and returns a new candidate grid (child)
def crossover(parent1, parent2):
    # Randomly selects a crossover point
    crossover_point = random.randint(1, GRID_SIZE - 1)
    # Returns child by concatenating parents after crossover points
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


# Mutated Grid Generator
# Takes candidate grid and returns random alterations by replacing values in cells not pre-filled in initial grid
def mutate(grid, initial_grid):
    # Creates deep copy to ensure mutations do not affect original grid
    mutated_grid = copy.deepcopy(grid)

    # Iterates over each cell in grid
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            # Looks for empty cells and mutates if randomly generated probabilty is less than MUTATION_RATE
            if initial_grid[i][j] is None and random.random() < MUTATION_RATE:
                # Assigns a random letter if conditions are met
                mutated_grid[i][j] = random.choice(LETTERS)
    # Returns mutated grid including random changes
    return mutated_grid


# Genetic Algorithm
# Takes the INITIAL_GRID and returns the best solution along with its fitness score
def genetic_algorithm(initial_grid):
   # Creates an initial population of candidate grids from the given initial_grid
    population = create_initial_population(initial_grid, POPULATION_SIZE)
    # Keeps track of best candidate grid
    best_solution = None
    # Keeps track of best fitness score - set to infinity, ensuring any solution will be better
    best_fitness = float('inf')

    # Loops for a predetermined number of GENERATIONS, processing each generation of candidate solutions
    for generation in range(GENERATIONS):
        # For each generation, sorts population by fitness (lower is better)
        population = sorted(population, key=calculate_fitness)
        # Chooses candidate with the lowest fitness score as current best
        current_best = population[0]
        current_fitness = calculate_fitness(current_best)
        # Prints current generation status and fitness score
        print(f"Generation {generation}: Best Fitness = {current_fitness}")

        # Updates best solution if current best solution in generation is better than any previous one
        if current_fitness < best_fitness:
            best_solution = current_best
            best_fitness = current_fitness
        # Terminates early if perfect solution is found
        if best_fitness == 0:
            break

        # If no perfect solution is found, creates new population through selection, crossover, and mutation
        # Variable to hold new population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            # Chooses 2 parent grids via tournament selection
            parent1, parent2 = selection(population)
            # Creates child grid by mixing the rows of the two parents
            child = crossover(parent1, parent2)
            # Potentially mutates child grid by randomly changing mutable cells
            child = mutate(child, initial_grid)
            new_population.append(child)
        population = new_population

    # Returns the best solution and its fitness score
    return best_solution, best_fitness


# Grid Printer
# Displays grid in readable format
def print_grid(grid):
    for row in grid:
        print(' '.join(row))


# Execution Mechanism
# Intiates Genetic Algorithm
def main():
    solution, fitness_score = genetic_algorithm(INITIAL_GRID)
    print("\nFinal Solution:")
    print_grid(solution)
    print(f"\nFinal Fitness Score: {fitness_score}")


# Determines whether the script is being run as the main program
if __name__ == '__main__':
    main()