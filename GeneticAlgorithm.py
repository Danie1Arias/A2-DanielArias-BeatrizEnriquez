import random

class GeneticAlgorithm:
    def __init__(self, data, population_size=100, generations=150, mutation_rate=0.2, elitism=0.1, selection_scheme='tournament'):
        self.data = data
        self.jobs = self.parse_data(data)
        self.num_jobs = len(self.jobs)
        self.num_tasks_per_job = len(self.jobs[0])
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.selection_scheme = selection_scheme

    def parse_data(self, data):
        jobs = []
        for row in data:
            job = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
            jobs.append(job)
        return jobs

    def generate_initial_solution(self, jobs):
        solution = []
        for job_id, job in enumerate(jobs):
            for task_id in range(len(job)):
                solution.append((job_id, task_id))
        random.shuffle(solution)
        return solution

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

    def swap_mutation(self, chromosome):
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return chromosome

    def inversion_mutation(self, chromosome):
        a, b = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[a:b] = reversed(chromosome[a:b])
        return chromosome

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

    def tournament_selection(self, population, fitnesses, k=3):
        selected = random.sample(list(zip(population, fitnesses)), k)
        selected.sort(key=lambda x: x[1])
        return selected[0][0]

    def roulette_wheel_selection(self, population, fitnesses):
        total_fitness = sum(1 / f for f in fitnesses)
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, fitness in zip(population, fitnesses):
            current += 1 / fitness
            if current > pick:
                return individual

    def rank_selection(self, population, fitnesses):
        ranked = sorted(zip(population, fitnesses), key=lambda x: x[1])
        ranks = [i + 1 for i in range(len(ranked))]
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        for i, (individual, _) in enumerate(ranked):
            current += ranks[i]
            if current > pick:
                return individual

    def stochastic_universal_sampling(self, population, fitnesses):
        total_fitness = sum(1 / f for f in fitnesses)
        distance = total_fitness / self.population_size
        start = random.uniform(0, distance)
        pointers = [start + i * distance for i in range(self.population_size)]
        current = 0
        selected = []

        for pointer in pointers:
            while current < len(fitnesses) and pointer > sum(1 / f for f in fitnesses[:current + 1]):
                current += 1
            selected.append(population[current])

        return selected

    def select(self, population, fitnesses):
        if self.selection_scheme == 'tournament':
            return self.tournament_selection(population, fitnesses)
        elif self.selection_scheme == 'roulette':
            return self.roulette_wheel_selection(population, fitnesses)
        elif self.selection_scheme == 'rank':
            return self.rank_selection(population, fitnesses)
        elif self.selection_scheme == 'sus':
            selected = self.stochastic_universal_sampling(population, fitnesses)
            return random.choice(selected)
        else:
            raise ValueError("Invalid selection scheme")

    def run(self):
        population = [self.generate_initial_solution(self.jobs) for _ in range(self.population_size)]
        best_solution = None
        best_makespan = float('inf')
        evolution = []
        elite_count = int(self.elitism * self.population_size)

        for generation in range(self.generations):
            fitnesses = [self.calculate_makespan(ind, self.jobs) for ind in population]

            current_best = min(fitnesses)
            if current_best < best_makespan:
                best_makespan = current_best
                best_solution = population[fitnesses.index(current_best)]

            evolution.append(best_makespan)

            sorted_population = [x for _, x in sorted(zip(fitnesses, population))]
            new_population = sorted_population[:elite_count]

            while len(new_population) < self.population_size:
                parent1 = self.select(population, fitnesses)
                parent2 = self.select(population, fitnesses)

                child1 = self.order_crossover(parent1, parent2)
                child2 = self.order_crossover(parent2, parent1)

                if random.random() < self.mutation_rate:
                    child1 = self.swap_mutation(child1) if random.random() < 0.5 else self.inversion_mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.swap_mutation(child2) if random.random() < 0.5 else self.inversion_mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]
            print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")

        decoded_solution = [(self.jobs[job_id][task_id]) for job_id, task_id in best_solution]
        return best_solution, best_makespan, evolution, decoded_solution