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

# Roulette Wheel Selection
def roulette_wheel_selection(population):
    # Calculate fitness for all candidates
    fitness_scores = [calculate_fitness(grid) for grid in population]
    
    # Find the maximum fitness score (worst solution)
    max_fitness = max(fitness_scores) + 1  # Add 1 to avoid division by zero
    
    # Convert fitness scores for minimization problem
    # Lower fitness (fewer conflicts) should get higher selection probability
    adjusted_scores = [max_fitness - score for score in fitness_scores]
    
    # Calculate total adjusted fitness
    total_fitness = sum(adjusted_scores)
    
    # Handle edge case where all solutions have the same fitness
    if total_fitness == 0:
        return random.sample(population, 2)
    
    # Calculate selection probabilities
    selection_probs = [score/total_fitness for score in adjusted_scores]
    
    # Select two parents
    parents = []
    
    for i in range(2):
        # Spin the wheel (generate random number)
        spin = random.random()
        current_sum = 0
        
        # Find the candidate where the spin lands
        for j, prob in enumerate(selection_probs):
            current_sum += prob
            if current_sum > spin:
                parents.append(population[j])
                break
    
    # In case of rounding errors leading to no selection
    if len(parents) < 2:
        missing = 2 - len(parents)
        additional_parents = random.sample(population, missing)
        parents.extend(additional_parents)
    
    return parents[0], parents[1]

# Crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(1, GRID_SIZE - 1)
    return parent1[:crossover_point] + parent2[crossover_point:]

# Mutation
def mutate(grid, initial_grid):
    mutated_grid = copy.deepcopy(grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if initial_grid[i][j] is None and random.random() < MUTATION_RATE:
                mutated_grid[i][j] = random.choice(LETTERS)
    return mutated_grid

# Genetic Algorithm with Selection Method Comparison
def genetic_algorithm(initial_grid):
    # Track results for both selection methods
    results = {
        'tournament': {'generations': [], 'time': []},
        'roulette': {'generations': [], 'time': []}
    }
    
    # Run multiple times to get statistical significance
    num_runs = 30
    
    for run in range(num_runs):
        print(f"Run {run+1}/{num_runs}:")
        
        # Create the initial population (same for both methods)
        initial_population = create_initial_population(initial_grid, POPULATION_SIZE)
        
        # Run with tournament selection
        start_time = time.time()
        tournament_gen = run_with_selection(initial_grid, initial_population.copy(), tournament_selection)
        tournament_time = time.time() - start_time
        results['tournament']['generations'].append(tournament_gen)
        results['tournament']['time'].append(tournament_time)
        
        # Run with roulette wheel selection
        start_time = time.time()
        roulette_gen = run_with_selection(initial_grid, initial_population.copy(), roulette_wheel_selection)
        roulette_time = time.time() - start_time
        results['roulette']['generations'].append(roulette_gen)
        results['roulette']['time'].append(roulette_time)
        
        print(f"  Tournament: {tournament_gen} generations, {tournament_time:.2f}s")
        print(f"  Roulette:   {roulette_gen} generations, {roulette_time:.2f}s")
    
    # Calculate and display averages
    avg_tournament_gen = sum(results['tournament']['generations']) / num_runs
    avg_tournament_time = sum(results['tournament']['time']) / num_runs
    avg_roulette_gen = sum(results['roulette']['generations']) / num_runs
    avg_roulette_time = sum(results['roulette']['time']) / num_runs
    
    print("\n=== Results Summary ===")
    print(f"Tournament Selection (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_tournament_gen:.2f}")
    print(f"  Average time: {avg_tournament_time:.2f}s")
    
    print(f"\nRoulette Wheel Selection (Average of {num_runs} runs):")
    print(f"  Average generations: {avg_roulette_gen:.2f}")
    print(f"  Average time: {avg_roulette_time:.2f}s")
    
    # Calculate improvement percentages
    if avg_roulette_gen > 0:
        gen_improvement = ((avg_roulette_gen - avg_tournament_gen) / avg_roulette_gen) * 100
        print(f"\nTournament selection is {gen_improvement:.2f}% faster in generations")
    
    if avg_roulette_time > 0:
        time_improvement = ((avg_roulette_time - avg_tournament_time) / avg_roulette_time) * 100
        print(f"Tournament selection is {time_improvement:.2f}% faster in time")
    
    return results

# Run the algorithm with a specific selection method
def run_with_selection(initial_grid, population, selection_func):
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
            parent1, parent2 = selection_func(population)
            child = crossover(parent1, parent2)
            child = mutate(child, initial_grid)
            new_population.append(child)
        
        population = new_population
    
    return GENERATIONS


# Main function
def main():
    print("Initial grid (puzzle to solve):")
    print_grid(INITIAL_GRID)
    print("\nComparing Tournament Selection vs. Roulette Wheel Selection")
    genetic_algorithm(INITIAL_GRID)

if __name__ == '__main__':
    main()