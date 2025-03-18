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

# Mutation method names
RANDOM_RESET = "random_reset"
SWAP = "swap"
SCRAMBLE = "scramble"

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

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, GRID_SIZE * GRID_SIZE - 1)
    
    # Flatten the grids for easier crossover
    flat_parent1 = [parent1[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    flat_parent2 = [parent2[i][j] for i in range(GRID_SIZE) for j in range(GRID_SIZE)]
    
    # Create child using crossover
    flat_child = flat_parent1[:crossover_point] + flat_parent2[crossover_point:]
    
    # Convert back to 2D grid
    child = []
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            row.append(flat_child[i * GRID_SIZE + j])
        child.append(row)
    
    return child

# Random Reset Mutation (Original method)
def random_reset_mutation(grid, initial_grid):
    mutated_grid = copy.deepcopy(grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if initial_grid[i][j] is None and random.random() < MUTATION_RATE:
                mutated_grid[i][j] = random.choice(LETTERS)
    return mutated_grid

# Swap Mutation
def swap_mutation(grid, initial_grid):
    mutated_grid = copy.deepcopy(grid)
    
    # Identify all mutable positions
    mutable_positions = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if initial_grid[i][j] is None:
                mutable_positions.append((i, j))
    
    # Apply swap mutation with probability MUTATION_RATE
    if random.random() < MUTATION_RATE and len(mutable_positions) >= 2:
        # Select two random positions to swap
        pos1, pos2 = random.sample(mutable_positions, 2)
        
        # Swap the values
        mutated_grid[pos1[0]][pos1[1]], mutated_grid[pos2[0]][pos2[1]] = \
        mutated_grid[pos2[0]][pos2[1]], mutated_grid[pos1[0]][pos1[1]]
    
    return mutated_grid

# Scramble Mutation
def scramble_mutation(grid, initial_grid):
    mutated_grid = copy.deepcopy(grid)
    
    # Apply scramble mutation with probability MUTATION_RATE
    if random.random() < MUTATION_RATE:
        # Randomly choose to scramble a row, column, or subgrid
        scramble_type = random.choice(['row', 'column', 'subgrid'])
        
        if scramble_type == 'row':
            # Select a random row
            row_idx = random.randint(0, GRID_SIZE - 1)
            
            # Identify mutable positions in this row
            mutable_indices = [j for j in range(GRID_SIZE) if initial_grid[row_idx][j] is None]
            
            if len(mutable_indices) >= 2:
                # Get current values at mutable positions
                values = [mutated_grid[row_idx][j] for j in mutable_indices]
                
                # Shuffle values
                random.shuffle(values)
                
                # Put shuffled values back
                for i, j in enumerate(mutable_indices):
                    mutated_grid[row_idx][j] = values[i]
                
        elif scramble_type == 'column':
            # Select a random column
            col_idx = random.randint(0, GRID_SIZE - 1)
            
            # Identify mutable positions in this column
            mutable_indices = [i for i in range(GRID_SIZE) if initial_grid[i][col_idx] is None]
            
            if len(mutable_indices) >= 2:
                # Get current values at mutable positions
                values = [mutated_grid[i][col_idx] for i in mutable_indices]
                
                # Shuffle values
                random.shuffle(values)
                
                # Put shuffled values back
                for i, row_idx in enumerate(mutable_indices):
                    mutated_grid[row_idx][col_idx] = values[i]
                
        else:  # subgrid
            # Select a random subgrid
            sub_i = random.randint(0, GRID_SIZE // SUBGRID_SIZE - 1) * SUBGRID_SIZE
            sub_j = random.randint(0, GRID_SIZE // SUBGRID_SIZE - 1) * SUBGRID_SIZE
            
            # Identify mutable positions in this subgrid
            mutable_positions = []
            for i in range(sub_i, sub_i + SUBGRID_SIZE):
                for j in range(sub_j, sub_j + SUBGRID_SIZE):
                    if initial_grid[i][j] is None:
                        mutable_positions.append((i, j))
            
            if len(mutable_positions) >= 2:
                # Get current values at mutable positions
                values = [mutated_grid[i][j] for i, j in mutable_positions]
                
                # Shuffle values
                random.shuffle(values)
                
                # Put shuffled values back
                for i, (row_idx, col_idx) in enumerate(mutable_positions):
                    mutated_grid[row_idx][col_idx] = values[i]
    
    return mutated_grid

# Mutation dispatcher
def mutate(grid, initial_grid, method):
    if method == RANDOM_RESET:
        return random_reset_mutation(grid, initial_grid)
    elif method == SWAP:
        return swap_mutation(grid, initial_grid)
    elif method == SCRAMBLE:
        return scramble_mutation(grid, initial_grid)
    else:
        raise ValueError(f"Unknown mutation method: {method}")

# Genetic Algorithm with one mutation method
def run_with_mutation(initial_grid, population, mutation_method):
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
            child = crossover(parent1, parent2)
            child = mutate(child, initial_grid, mutation_method)
            new_population.append(child)
        
        population = new_population
    
    return GENERATIONS

# Genetic Algorithm with Mutation Method Comparison
def genetic_algorithm(initial_grid):
    # Track results for each mutation method
    results = {
        RANDOM_RESET: {'generations': [], 'time': []},
        SWAP: {'generations': [], 'time': []},
        SCRAMBLE: {'generations': [], 'time': []}
    }
    
    # Run multiple times to get statistical significance
    num_runs = 30
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}:")
        
        # Create the initial population (same for all methods)
        initial_population = create_initial_population(initial_grid, POPULATION_SIZE)
        
        # Run with random reset mutation
        random_reset_gen = run_with_mutation(initial_grid, initial_population.copy(), RANDOM_RESET)
        results[RANDOM_RESET]['generations'].append(random_reset_gen)
        
        # Run with swap mutation
        swap_gen = run_with_mutation(initial_grid, initial_population.copy(), SWAP)
        results[SWAP]['generations'].append(swap_gen)
        
        # Run with scramble mutation
        scramble_gen = run_with_mutation(initial_grid, initial_population.copy(), SCRAMBLE)
        results[SCRAMBLE]['generations'].append(scramble_gen)
        
        print(f"  Random Reset: {random_reset_gen} generations")
        print(f"  Swap:         {swap_gen} generations")
        print(f"  Scramble:     {scramble_gen} generations")
    
    # Calculate and display averages
    avg_random_reset_gen = sum(results[RANDOM_RESET]['generations']) / num_runs
    avg_swap_gen = sum(results[SWAP]['generations']) / num_runs
    avg_scramble_gen = sum(results[SCRAMBLE]['generations']) / num_runs
    
    print("\n=== Results Summary ===")
    print(f"Random Reset Mutation (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_random_reset_gen:.2f}")
    
    print(f"\nSwap Mutation (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_swap_gen:.2f}")
    
    print(f"\nScramble Mutation (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_scramble_gen:.2f}")
    
    # Find the best method
    methods = {
        'Random Reset': avg_random_reset_gen,
        'Swap': avg_swap_gen,
        'Scramble': avg_scramble_gen
    }
    best_method = min(methods, key=methods.get)
    print(f"\nBest method by generations: {best_method}")
    
    return results

# Main function
def main():
    print("=== Letter Sudoku Solver with Mutation Method Comparison ===")
    print("Initial grid (puzzle to solve):")
    print_grid(INITIAL_GRID)
    print("\nComparing Random Reset, Swap, and Scramble Mutation...")
    genetic_algorithm(INITIAL_GRID)

if __name__ == '__main__':
    main()