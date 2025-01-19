import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
        job_completion_time = [0] * len(jobs)
        task_schedule = []

        for job_id, task_id in chromosome:
            machine, duration = jobs[job_id][task_id]
            earliest_start_time = job_completion_time[job_id]
            start_time = max(machine_time[machine], earliest_start_time)
            end_time = start_time + duration
            machine_time[machine] = end_time
            job_completion_time[job_id] = end_time

            task_schedule.append({
                'machine': machine,
                'start': start_time,
                'end': end_time,
                'name': f'job({job_id}, {task_id})'
            })

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
            # print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")

        makespan, best_schedule = self.calculate_makespan(best_solution, self.jobs)
        return best_solution, best_makespan, evolution, best_schedule

    def plot_results(self, evolution, schedule):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].plot(evolution, color='blue')
        axs[0].set_title('Evolution of Makespan', fontsize=14)
        axs[0].set_xlabel('Generation', fontsize=12)
        axs[0].set_ylabel('Makespan', fontsize=12)
        axs[0].grid(True, linestyle='--', alpha=0.7)

        task_colors = plt.cm.tab20.colors
        for task in schedule:
            machine = task['machine']
            start = task['start']
            end = task['end']
            name = task['name']
            job_id = int(name.split('(')[1].split(',')[0])
            color = task_colors[job_id % len(task_colors)]

            axs[1].broken_barh([(start, end - start)], (machine - 0.4, 0.8), facecolors=color, edgecolor='black', alpha=0.8)
            axs[1].text(x=start + (end - start) / 2, y=machine, s=name, va='center', ha='center', color='white', fontsize=8, weight='bold', clip_on=True)

        machines = sorted(set(task['machine'] for task in schedule))
        axs[1].set_yticks(machines)
        axs[1].set_yticklabels([f'Machine {i}' for i in machines], fontsize=10)
        axs[1].set_title('Gantt Chart for the Final Solution', fontsize=14)
        axs[1].set_xlabel('Time', fontsize=12)
        axs[1].set_ylabel('Machines', fontsize=12)
        axs[1].grid(True, axis='x', linestyle='--', alpha=0.7)

        job_ids = sorted(set(int(task['name'].split('(')[1].split(',')[0]) for task in schedule))
        patches = [mpatches.Patch(color=task_colors[job_id % len(task_colors)], label=f'Job {job_id}') for job_id in job_ids]
        axs[1].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

        plt.tight_layout()
        plt.show()
