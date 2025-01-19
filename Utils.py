import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def plot_gantt_chart(schedule):
    task_colors = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(10, 6))

    for task in schedule:
        machine = task['machine']
        start = task['start']
        end = task['end']
        name = task['name']
        job_id = int(name.split('(')[1].split(',')[0])
        color = task_colors[job_id % len(task_colors)]
        ax.broken_barh(
            [(start, end - start)],
            (machine - 0.4, 0.8),
            facecolors=color,
            edgecolor='black',
        )

    # Ensure the y-axis ticks and labels align
    machines = sorted(set(task['machine'] for task in schedule))
    ax.set_yticks(machines)
    ax.set_yticklabels([f'Machine {i}' for i in machines])

    ax.set_xlabel('Time')
    ax.set_ylabel('Machines')
    ax.set_title('Gantt Chart for the Final Solution')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # Add a legend for jobs
    job_ids = sorted(set(int(task['name'].split('(')[1].split(',')[0]) for task in schedule))
    patches = [
        mpatches.Patch(color=task_colors[job_id % len(task_colors)], label=f'Job {job_id}')
        for job_id in job_ids
    ]
    ax.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    plt.show()

    # Print task details for clarity
    print("\nTask Details:")
    for task in sorted(schedule, key=lambda x: (x['machine'], x['start'])):
        print(
            f"Machine {task['machine']}: {task['name']}, "
            f"Start Time: {task['start']}, End Time: {task['end']}"
        )
