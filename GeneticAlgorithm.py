import random
import numpy as np
import matplotlib.pyplot as plt

# Define the new input data (Adams, Balas, and Zawack 10x10 instance, instance 5)
data = [
    [4, 88, 8, 68, 6, 94, 5, 99, 1, 67],
    [2, 89, 9, 77, 7, 99, 0, 86, 3, 92],
    [5, 72, 3, 50, 6, 69, 4, 75, 2, 94],
    [8, 66, 0, 92, 1, 82, 7, 94, 9, 63],
    [9, 83, 8, 61, 0, 83, 1, 65, 6, 64],
    [5, 85, 7, 78, 4, 85, 2, 55, 3, 77],
    [7, 94, 2, 68, 1, 61, 4, 99, 3, 54],
    [6, 75, 5, 66, 0, 76, 9, 63, 8, 67],
    [3, 69, 4, 88, 9, 82, 8, 95, 0, 99],
    [2, 67, 6, 95, 5, 68, 7, 67, 1, 86]
]

num_jobs = len(data)
num_machines = len(data[0])

# Convert the input data to the format [(machine, time)] for each job
jobs = []
for row in data:
    job = [(i, time) for i, time in enumerate(row)]
    jobs.append(job)

# Generate a valid initial solution
def generate_initial_solution(jobs):
    solution = []
    for job_id, job in enumerate(jobs):
        for task_id in range(len(job)):
            solution.append((job_id, task_id))
    random.shuffle(solution)
    return solution

# Decode a chromosome and calculate the makespan
def calculate_makespan(chromosome, jobs):
    machine_time = [0] * num_machines
    job_time = [0] * num_jobs

    for job_id, task_id in chromosome:
        machine, duration = jobs[job_id][task_id]
        start_time = max(machine_time[machine], job_time[job_id])
        machine_time[machine] = start_time + duration
        job_time[job_id] = start_time + duration

    return max(machine_time)

# Perform mutation (swap mutation)
def swap_mutation(chromosome):
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome

# Perform inversion mutation
def inversion_mutation(chromosome):
    a, b = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[a:b] = reversed(chromosome[a:b])
    return chromosome

# Perform crossover (Order Crossover - OX1)
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    # Copy segment from parent1
    child[start:end] = parent1[start:end]

    # Fill the remaining positions from parent2
    pointer = end
    for gene in parent2:
        if gene not in child:
            if pointer >= size:
                pointer = 0
            child[pointer] = gene
            pointer += 1

    return child

# Tournament selection
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1])
    return selected[0][0]

# Genetic Algorithm for JSSP
def genetic_algorithm(jobs, population_size=100, generations=150, mutation_rate=0.2, elitism=0.1):
    # Initial population
    population = [generate_initial_solution(jobs) for _ in range(population_size)]
    best_solution = None
    best_makespan = float('inf')
    evolution = []
    elite_count = int(elitism * population_size)

    for generation in range(generations):
        fitnesses = [calculate_makespan(ind, jobs) for ind in population]

        # Record best solution
        current_best = min(fitnesses)
        if current_best < best_makespan:
            best_makespan = current_best
            best_solution = population[fitnesses.index(current_best)]

        evolution.append(best_makespan)

        # Elitism: Preserve the best individuals
        sorted_population = [x for _, x in sorted(zip(fitnesses, population))]
        new_population = sorted_population[:elite_count]

        # Generate new population
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1 = order_crossover(parent1, parent2)
            child2 = order_crossover(parent2, parent1)

            if random.random() < mutation_rate:
                child1 = swap_mutation(child1) if random.random() < 0.5 else inversion_mutation(child1)
            if random.random() < mutation_rate:
                child2 = swap_mutation(child2) if random.random() < 0.5 else inversion_mutation(child2)

            new_population.extend([child1, child2])

        population = new_population[:population_size]
        print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")

    # Plot evolution
    plt.plot(evolution, label='Best Makespan')
    plt.xlabel('Generation')
    plt.ylabel('Makespan')
    plt.title('Evolution of Makespan Over Generations')
    plt.legend()
    plt.show()

    return best_solution, best_makespan

# Run the genetic algorithm
best_solution, best_makespan = genetic_algorithm(jobs)
print("Best Solution:", best_solution)
print("Best Makespan:", best_makespan)
