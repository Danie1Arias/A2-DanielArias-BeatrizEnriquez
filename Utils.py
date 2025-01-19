import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Parse the data into a list of jobs
def parse_data(data):
    jobs = []
    for row in data:
        job = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
        jobs.append(job)
    return jobs

# Generate an initial solution by randomly shuffling the tasks
def generate_initial_solution(jobs):
    solution = []
    for job_id, job in enumerate(jobs):
        for task_id in range(len(job)):
            solution.append((job_id, task_id))
    random.shuffle(solution)
    return solution

# Calculate the makespan of a chromosome and the schedule of tasks
def calculate_makespan(chromosome, jobs):
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

# Plot the evolution of the best makespan and the final schedule
def plot_results(evolution, schedule):
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
    axs[1].set_title('Gantt Chart', fontsize=14)
    axs[1].set_xlabel('Time', fontsize=12)
    axs[1].set_ylabel('Machines', fontsize=12)
    axs[1].grid(True, axis='x', linestyle='--', alpha=0.7)

    job_ids = sorted(set(int(task['name'].split('(')[1].split(',')[0]) for task in schedule))
    patches = [mpatches.Patch(color=task_colors[job_id % len(task_colors)], label=f'Job {job_id}') for job_id in job_ids]
    axs[1].legend(handles=patches, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)

    plt.tight_layout()
    plt.show()
    