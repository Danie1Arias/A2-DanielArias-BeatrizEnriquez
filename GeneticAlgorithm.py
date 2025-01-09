import random

class GeneticAlgorithm:
# Datos de entrada (Adams, Balas, y Zawack 10x10 instancia 5)
    def __init__(self, data):
        self.data = data
        self.jobs = self.parse_data(data)
        self.num_jobs = len(self.jobs)
        self.num_tasks_per_job = len(self.jobs[0])

    # Función para procesar los datos
    def parse_data(self, data):
        jobs = []
        for row in data:
            job = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
            jobs.append(job)
        return jobs

    # Generar una solución inicial válida
    def generate_initial_solution(self, jobs):
        solution = []
        for job_id, job in enumerate(jobs):
            for task_id in range(len(job)):
                solution.append((job_id, task_id))
        random.shuffle(solution)
        return solution

    # Decodificar un cromosoma y calcular el makespan
    def calculate_makespan(self, chromosome, jobs):
        num_machines = max(max(machine for machine, _ in job) for job in jobs) + 1
        machine_time = [0] * num_machines
        job_time = [0] * self.num_jobs

        for job_id, task_id in chromosome:
            machine, duration = jobs[job_id][task_id]
            start_time = max(machine_time[machine], job_time[job_id])
            machine_time[machine] = start_time + duration
            job_time[job_id] = start_time + duration

        return max(machine_time)

    # Mutación por intercambio
    def swap_mutation(self, chromosome):
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return chromosome

    # Mutación por inversión
    def inversion_mutation(self, chromosome):
        a, b = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[a:b] = reversed(chromosome[a:b])
        return chromosome

    # Cruce (Order Crossover - OX1)
    def order_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        child = [None] * size

        child[start:end] = parent1[start:end]

        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer >= size:
                    pointer = 0
                child[pointer] = gene
                pointer += 1

        return child

    # Selección por torneo
    def tournament_selection(self, population, fitnesses, k=3):
        selected = random.sample(list(zip(population, fitnesses)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    # Algoritmo genético
    def genetic_algorithm(self, population_size=100, generations=150, mutation_rate=0.2, elitism=0.1):
        population = [self.generate_initial_solution(self.jobs) for _ in range(population_size)]
        best_solution = None
        best_makespan = float('inf')
        evolution = []
        elite_count = int(elitism * population_size)

        for generation in range(generations):
            fitnesses = [self.calculate_makespan(ind, self.jobs) for ind in population]

            current_best = min(fitnesses)
            if current_best < best_makespan:
                best_makespan = current_best
                best_solution = population[fitnesses.index(current_best)]

            evolution.append(best_makespan)

            sorted_population = [x for _, x in sorted(zip(fitnesses, population))]
            new_population = sorted_population[:elite_count]

            while len(new_population) < population_size:
                parent1 = self.tournament_selection(population, fitnesses)
                parent2 = self.tournament_selection(population, fitnesses)
                child1 = self.order_crossover(parent1, parent2)
                child2 = self.order_crossover(parent2, parent1)

                if random.random() < mutation_rate:
                    child1 = self.swap_mutation(child1) if random.random() < 0.5 else self.inversion_mutation(child1)
                if random.random() < mutation_rate:
                    child2 = self.swap_mutation(child2) if random.random() < 0.5 else self.inversion_mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:population_size]
            print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")

        decoded_solution = [(self.jobs[job_id][task_id]) for job_id, task_id in best_solution]
        return best_solution, best_makespan, evolution, decoded_solution 
