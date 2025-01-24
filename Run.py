from GeneticAlgorithm import GeneticAlgorithm
from Utils import plot_results

data = [
    [0, 3, 1, 2, 2, 2],
    [0, 2, 2, 1, 1, 4],
    [1, 4, 2, 3]
]

ga = GeneticAlgorithm(data, population_size=3, generations=50, mutation_rate=0.3, elitism=0.1, 
                      selection_scheme='tournament', crossover_scheme='one_point', mutation_scheme='one_mutation')
best_solution, best_makespan, evolution, schedule = ga.run()

print("Best Makespan:", best_makespan)
plot_results(evolution, schedule)