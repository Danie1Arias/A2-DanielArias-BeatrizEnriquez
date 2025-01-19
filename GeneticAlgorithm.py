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
        # Initialize variables
        num_machines = max(max(machine for machine, _ in job) for job in jobs) + 1
        machine_time = [0] * num_machines  # Tracks when each machine becomes available
        job_completion_time = [0] * len(jobs)  # Tracks when the last task of each job is completed
        task_schedule = []  # Stores the task scheduling details

        # Loop through the chromosome to schedule tasks
        for job_id, task_id in chromosome:
            machine, duration = jobs[job_id][task_id]

            # Ensure the task starts only after the previous task in the same job is completed
            earliest_start_time = job_completion_time[job_id]

            # Ensure the machine is available before the task starts
            start_time = max(machine_time[machine], earliest_start_time)
            end_time = start_time + duration

            # Update machine and job completion times
            machine_time[machine] = end_time
            job_completion_time[job_id] = end_time

            # Add the task to the schedule
            task_schedule.append({
                'machine': machine,
                'start': start_time,
                'end': end_time,
                'name': f'job({job_id}, {task_id})'
            })

        # Makespan is the maximum time when all machines finish their tasks
        makespan = max(machine_time)
        return makespan, task_schedule



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

    def select(self, population, fitnesses):
        if self.selection_scheme == 'tournament':
            return self.tournament_selection(population, fitnesses)
        else:
            raise ValueError("Invalid selection scheme")

    def run(self):
        population = [self.generate_initial_solution(self.jobs) for _ in range(self.population_size)]
        best_solution = None
        best_makespan = float('inf')
        evolution = []
        elite_count = int(self.elitism * self.population_size)

        for generation in range(self.generations):
            fitnesses_and_schedules = [self.calculate_makespan(ind, self.jobs) for ind in population]
            fitnesses = [fs[0] for fs in fitnesses_and_schedules]
            schedules = [fs[1] for fs in fitnesses_and_schedules]

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

        makespan, best_schedule = self.calculate_makespan(best_solution, self.jobs)
        return best_solution, best_makespan, evolution, best_schedule
