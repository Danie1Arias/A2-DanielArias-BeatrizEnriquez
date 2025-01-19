from GeneticAlgorithm import GeneticAlgorithm
from Utils import plot_gantt_chart
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

data = [
    [0, 3, 1, 2, 2, 2],
    [0, 2, 2, 1, 1, 4],
    [1, 4, 2, 3]
]

# Run the Genetic Algorithm
ga = GeneticAlgorithm(data, population_size=150, generations=200, mutation_rate=0.1, elitism=0.1)
best_solution, best_makespan, evolution, schedule = ga.run()

print("Best Makespan:", best_makespan)

# Create a figure with two subplots
fig, axs = plt.subplots(1, 2, figsize=(18, 8))

# Plot the evolution of makespan on the first subplot
axs[0].plot(evolution, color='blue')
axs[0].set_title('Evolution of Makespan', fontsize=14)
axs[0].set_xlabel('Generation', fontsize=12)
axs[0].set_ylabel('Makespan', fontsize=12)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Plot the Gantt chart on the second subplot
task_colors = plt.cm.tab20.colors
for task in schedule:
    machine = task['machine']
    start = task['start']
    end = task['end']
    name = task['name']
    job_id = int(name.split('(')[1].split(',')[0])
    color = task_colors[job_id % len(task_colors)]
    
    # Add the task bar
    axs[1].broken_barh(
        [(start, end - start)],
        (machine - 0.4, 0.8),
        facecolors=color,
        edgecolor='black',
        alpha=0.8
    )
    
    # Add labels to each task
    axs[1].text(
        x=start + (end - start) / 2,
        y=machine,
        s=name,
        va='center',
        ha='center',
        color='white',
        fontsize=8,
        weight='bold',
        clip_on=True
    )

# Set up Gantt chart subplot
machines = sorted(set(task['machine'] for task in schedule))
axs[1].set_yticks(machines)
axs[1].set_yticklabels([f'Machine {i}' for i in machines], fontsize=10)
axs[1].set_title('Gantt Chart for the Final Solution', fontsize=14)
axs[1].set_xlabel('Time', fontsize=12)
axs[1].set_ylabel('Machines', fontsize=12)
axs[1].grid(True, axis='x', linestyle='--', alpha=0.7)

# Add a legend for jobs
job_ids = sorted(set(int(task['name'].split('(')[1].split(',')[0]) for task in schedule))
patches = [
    mpatches.Patch(color=task_colors[job_id % len(task_colors)], label=f'Job {job_id}')
    for job_id in job_ids
]
axs[1].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

# Adjust layout and display
plt.tight_layout()
plt.show()
