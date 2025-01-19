from GeneticAlgorithm import GeneticAlgorithm
from Utils import plot_gantt_chart
import matplotlib.pyplot as plt

data = [
    [0, 3, 1, 2, 2, 2],
    [0, 2, 2, 1, 1, 4],
    [1, 4, 2, 3]
]

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

ga = GeneticAlgorithm(data, population_size=3, generations=20, mutation_rate=0.1, elitism=0.1)
best_solution, best_makespan, evolution, schedule = ga.run()

print("Best Makespan:", best_makespan)
print("Schedule:", schedule)

plt.plot(evolution)
plt.title('Evolution of Makespan')
plt.xlabel('Generation')
plt.ylabel('Makespan')
plt.show()

plot_gantt_chart(schedule)