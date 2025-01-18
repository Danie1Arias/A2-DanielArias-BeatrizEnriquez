import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt_chart(decoded_solution):
    """
    Plot a Gantt chart based on the decoded solution.
    
    Parameters:
        decoded_solution (list of tuples): List where each tuple contains (machine, start_time, duration).
    """
    # Prepare data for Gantt chart
    tasks_per_machine = {}
    current_time_per_machine = {}

    for machine, duration in decoded_solution:
        start_time = current_time_per_machine.get(machine, 0)
        tasks_per_machine.setdefault(machine, []).append((start_time, duration))
        current_time_per_machine[machine] = start_time + duration

    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10.colors  # Use a color map for better distinction

    for machine, tasks in tasks_per_machine.items():
        for task_id, (start_time, duration) in enumerate(tasks):
            ax.broken_barh([(start_time, duration)], (machine - 0.4, 0.8), 
                           facecolors=colors[task_id % len(colors)], edgecolor="black")

    ax.set_yticks(range(len(tasks_per_machine)))
    ax.set_yticklabels([f"Machine {i}" for i in tasks_per_machine.keys()])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_title("Gantt Chart for the Final Solution")
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add legend
    patches = [mpatches.Patch(color=colors[i % len(colors)], label=f"Task {i}") for i in range(len(decoded_solution))]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.show()
