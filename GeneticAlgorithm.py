import random
from Utils import parse_data, generate_initial_solution, calculate_makespan, order_chromosome

class GeneticAlgorithm:
    def __init__(self, data, population_size=100, generations=150, mutation_rate=0.2, elitism=0.1, 
                 selection_scheme='tournament', crossover_scheme='order', mutation_scheme='one_mutation'):
        self.data = data
        self.jobs = parse_data(self.data)
        self.num_jobs = len(self.jobs)
        self.num_tasks_per_job = len(self.jobs[0])
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.selection_scheme = selection_scheme
        self.crossover_scheme = crossover_scheme
        self.mutation_scheme = mutation_scheme

    # Crossover functions
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
    
    def one_point_crossover(self, parent1, parent2):
        size = len(parent1)
        point = random.randint(1, size - 1)

        # Exchange the tails of the parents
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]

        # Exchange duplicated genes in child1 and child2
        for i in range(point, size):
            if child1[i] in child1[:i]:
                child1[i] = next(gene for gene in parent2 if gene not in child1)
            if child2[i] in child2[:i]:
                child2[i] = next(gene for gene in parent1 if gene not in child2)
        
        return order_chromosome(child1), order_chromosome(child2)
    
    def two_point_crossover(self, parent1, parent2):
        size = len(parent1)
        point1, point2 = sorted(random.sample(range(size), 2))
        child = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
        return child
    
    def uniform_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [None] * size
        for i in range(size):
            if random.random() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
    
    def crossover(self, parent1, parent2):
        if self.crossover_scheme == 'order':
            return self.order_crossover(parent1, parent2)
        elif self.crossover_scheme == 'one_point':
            return self.one_point_crossover(parent1, parent2)
        elif self.crossover_scheme == 'two_point':
            return self.two_point_crossover(parent1, parent2)
        elif self.crossover_scheme == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        else:
            raise ValueError("Invalid crossover scheme")

    # Selection functions
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
            if current >= pick:
                return individual

    def rank_selection(self, population, fitnesses):
        sorted_population = [x for _, x in sorted(zip(fitnesses, population))]
        ranks = range(1, len(sorted_population) + 1)
        total_rank = sum(ranks)
        pick = random.uniform(0, total_rank)
        current = 0
        for rank, individual in zip(ranks, sorted_population):
            current += rank
            if current >= pick:
                return individual

    def stochastic_universal_sampling(self, population, fitnesses):
        total_fitness = sum(1 / f for f in fitnesses)
        distance = total_fitness / len(population)
        start = random.uniform(0, distance)
        points = [start + i * distance for i in range(len(population))]
        individuals = []
        for point in points:
            current = 0
            for individual, fitness in zip(population, fitnesses):
                current += 1 / fitness
                if current >= point:
                    individuals.append(individual)
                    break
        return random.choice(individuals)

    def select(self, population, fitnesses):
        if self.selection_scheme == 'tournament':
            return self.tournament_selection(population, fitnesses)
        elif self.selection_scheme == 'roulette':
            return self.roulette_wheel_selection(population, fitnesses)
        elif self.selection_scheme == 'rank':
            return self.rank_selection(population, fitnesses)
        elif self.selection_scheme == 'sus':
            return self.stochastic_universal_sampling(population, fitnesses)
        else:
            raise ValueError("Invalid selection scheme")

    # Mutation functions       
    def swap_mutation(self, chromosome):
        a, b = random.sample(range(len(chromosome)), 2)
        chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
        return chromosome

    def inversion_mutation(self, chromosome):
        a, b = sorted(random.sample(range(len(chromosome)), 2))
        chromosome[a:b] = reversed(chromosome[a:b])
        return chromosome
        
    def mutate(self, chromosome):
        if self.mutation_scheme == 'one_mutation':
            if random.random() < 0.5:
                return self.swap_mutation(chromosome)
            else:
                return self.inversion_mutation(chromosome)
        
        elif self.mutation_scheme == 'multi_mutation':
            num_mutations = random.randint(1, 3)
            for _ in range(num_mutations):
                if random.random() < 0.5:
                    chromosome = self.swap_mutation(chromosome)
                else:
                    chromosome = self.inversion_mutation(chromosome)
            return chromosome
        
        else:
            raise ValueError("Invalid mutation scheme")

    # Main function to run the genetic algorithm
    def run(self):
        population = [generate_initial_solution(self.jobs) for _ in range(self.population_size)]
        best_solution = None
        best_makespan = float('inf')
        evolution = []
        elite_count = int(self.elitism * self.population_size)

        for generation in range(self.generations):
            fitnesses_and_schedules = [calculate_makespan(ind, self.jobs) for ind in population]
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
                # Selection
                parent1 = self.select(population, fitnesses)
                parent2 = self.select(population, fitnesses)
                
                # Crossover
                children = self.crossover(parent1, parent2)
                child1 = children[0]
                child2 = children[1]

                # Mutation
                # child1 = self.mutate(child1) if random.random() < self.mutation_rate else child1
                # child2 = self.mutate(child2) if random.random() < self.mutation_rate else child2

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]
            # print(f"Generation {generation + 1}: Best Makespan = {best_makespan}")
        makespan, best_schedule = calculate_makespan(best_solution, self.jobs)
        return best_solution, best_makespan, evolution, best_schedule
