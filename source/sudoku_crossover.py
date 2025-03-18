import random
import copy
import time

# Configuration Constants
LETTERS = ['W', 'D', 'R', 'O']
GRID_SIZE = 4
SUBGRID_SIZE = 2
POPULATION_SIZE = 100
GENERATIONS = 1000
MUTATION_RATE = 0.1
INITIAL_GRID = [
    ['W', None, None, 'O'],
    [None, 'R', None, None],
    [None, None, 'D', None],
    ['D', None, None, 'R']
]

# Grid Printer
def print_grid(grid):
    for row in grid:
        formatted_row = [cell if cell is not None else '_' for cell in row]
        print(' '.join(formatted_row))

# Random Grid Generator
def initialize_grid(initial_grid):
    grid = copy.deepcopy(initial_grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if grid[i][j] is None:
                grid[i][j] = random.choice(LETTERS)
    return grid

# Population Generator
def create_initial_population(initial_grid, population_size):
    population = []
    for _ in range(population_size):
        candidate = initialize_grid(initial_grid)
        population.append(candidate)
    return population

# Fitness Score Calculator
def calculate_fitness(grid):
    conflicts = 0
    
    # Row conflicts
    for row in grid:
        conflicts += GRID_SIZE - len(set(row))
    
    # Column conflicts
    for j in range(GRID_SIZE):
        col = [grid[i][j] for i in range(GRID_SIZE)]
        conflicts += GRID_SIZE - len(set(col))
    
    # Subgrid conflicts
    for sub_i in range(0, GRID_SIZE, SUBGRID_SIZE):
        for sub_j in range(0, GRID_SIZE, SUBGRID_SIZE):
            block = []
            for i in range(sub_i, sub_i + SUBGRID_SIZE):
                for j in range(sub_j, sub_j + SUBGRID_SIZE):
                    block.append(grid[i][j])
            conflicts += GRID_SIZE - len(set(block))
    
    return conflicts

# Tournament Selection
def tournament_selection(population):
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    tournament.sort(key=calculate_fitness)
    return tournament[0], tournament[1]

# Single-Point Crossover
def single_point_crossover(parent1, parent2):
    crossover_point = random.randint(1, GRID_SIZE - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Two-Point Crossover
def two_point_crossover(parent1, parent2):
    # Ensure point1 < point2
    point1 = random.randint(1, GRID_SIZE - 2)
    point2 = random.randint(point1 + 1, GRID_SIZE - 1)
    
    # Take beginning from parent1, middle from parent2, end from parent1
    child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    return child

# Uniform Crossover
def uniform_crossover(parent1, parent2):
    child = []
    for i in range(GRID_SIZE):
        # Create a new row
        new_row = []
        for j in range(GRID_SIZE):
            # For each cell, randomly choose from either parent
            if random.random() < 0.5:
                new_row.append(parent1[i][j])
            else:
                new_row.append(parent2[i][j])
        child.append(new_row)
    return child

# Mutation
def mutate(grid, initial_grid):
    mutated_grid = copy.deepcopy(grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if initial_grid[i][j] is None and random.random() < MUTATION_RATE:
                mutated_grid[i][j] = random.choice(LETTERS)
    return mutated_grid

# Genetic Algorithm with Crossover Method Comparison
def genetic_algorithm(initial_grid):
    # Track results for each crossover method
    results = {
        'single_point': {'generations': [], 'time': []},
        'two_point': {'generations': [], 'time': []},
        'uniform': {'generations': [], 'time': []}
    }
    
    # Run multiple times to get statistical significance
    num_runs = 30
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}:")
        
        # Create the initial population (same for all methods)
        initial_population = create_initial_population(initial_grid, POPULATION_SIZE)
        
        # Run with single-point crossover
        single_point_gen = run_with_crossover(initial_grid, initial_population.copy(), single_point_crossover)
        results['single_point']['generations'].append(single_point_gen)
        
        # Run with two-point crossover
        two_point_gen = run_with_crossover(initial_grid, initial_population.copy(), two_point_crossover)
        results['two_point']['generations'].append(two_point_gen)
        
        # Run with uniform crossover
        uniform_gen = run_with_crossover(initial_grid, initial_population.copy(), uniform_crossover)
        results['uniform']['generations'].append(uniform_gen)
        
        print(f"  Single-point: {single_point_gen} generations")
        print(f"  Two-point:    {two_point_gen} generations")
        print(f"  Uniform:      {uniform_gen} generations")
    
    # Calculate and display averages
    avg_single_point_gen = sum(results['single_point']['generations']) / num_runs
    avg_two_point_gen = sum(results['two_point']['generations']) / num_runs
    avg_uniform_gen = sum(results['uniform']['generations']) / num_runs
    
    print("\n=== Results Summary ===")
    print(f"Single-point Crossover (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_single_point_gen:.2f}")
    
    print(f"\nTwo-point Crossover (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_two_point_gen:.2f}")
    
    print(f"\nUniform Crossover (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_uniform_gen:.2f}")
    
    # Find the best method
    methods = {
        'Single-point': avg_single_point_gen,
        'Two-point': avg_two_point_gen,
        'Uniform': avg_uniform_gen
    }
    best_method = min(methods, key=methods.get)
    print(f"\nBest method by generations: {best_method}")
    
    return results

# Run the algorithm with a specific crossover method
def run_with_crossover(initial_grid, population, crossover_func):
    best_fitness = float('inf')
    
    for generation in range(GENERATIONS):
        population = sorted(population, key=calculate_fitness)
        current_best = population[0]
        current_fitness = calculate_fitness(current_best)
        
        if current_fitness < best_fitness:
            best_fitness = current_fitness
        
        if best_fitness == 0:
            return generation + 1
        
        new_population = []
        
        while len(new_population) < POPULATION_SIZE:
            parent1, parent2 = tournament_selection(population)
            child = crossover_func(parent1, parent2)
            child = mutate(child, initial_grid)
            new_population.append(child)
        
        population = new_population
    
    return GENERATIONS

# Main function
def main():
    print("=== Letter Sudoku Solver with Crossover Method Comparison ===")
    print("Initial grid (puzzle to solve):")
    print_grid(INITIAL_GRID)
    print("\nComparing Single-point, Two-point, and Uniform Crossover...")
    genetic_algorithm(INITIAL_GRID)

if __name__ == '__main__':
    main()
