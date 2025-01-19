import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class TabuSearch:
    def __init__(self, data, max_iterations=1000, tabu_tenure=10, aspiration_criteria=True):
        self.data = data
        self.jobs = self.parse_data(data)
        self.num_jobs = len(self.jobs)
        self.num_tasks_per_job = len(self.jobs[0])
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_list = {}
        self.best_solution = None
        self.best_makespan = float('inf')
        self.aspiration_criteria = aspiration_criteria

    # Parse the data into a list of jobs
    def parse_data(self, data):
        jobs = []
        for row in data:
            job = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
            jobs.append(job)
        return jobs

    # Generate an initial random solution (random order of tasks across all jobs)
    def generate_initial_solution(self):
        solution = []
        for job_id, job in enumerate(self.jobs):
            for task_id in range(len(job)):
                solution.append((job_id, task_id))
        random.shuffle(solution)
        return solution

    # Calculate makespan of a given solution
    def calculate_makespan(self, solution):
        num_machines = max(max(machine for machine, _ in job) for job in self.jobs) + 1
        machine_time = [0] * num_machines
        job_completion_time = [0] * len(self.jobs)
        task_schedule = []

        for job_id, task_id in solution:
            machine, duration = self.jobs[job_id][task_id]
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

    # Perform a swap operation on the solution (neighbouring move)
    def swap_move(self, solution):
        # Convert tuple to list to make it mutable
        new_solution = list(solution)
        a, b = random.sample(range(len(solution)), 2)
        new_solution[a], new_solution[b] = new_solution[b], new_solution[a]
        
        # Convert list back to tuple for immutability and hashing
        new_solution_tuple = tuple(new_solution)
        return new_solution_tuple

    # Check if a move is in the tabu list
    def is_tabu(self, move):
        # Convert list of tuples (move) to tuple of tuples to make it hashable
        move_tuple = tuple(move)
        return move_tuple in self.tabu_list and self.tabu_list[move_tuple] > 0

    # Update the tabu list by decrementing the tenure of each entry
    def update_tabu_list(self):
        for move in list(self.tabu_list.keys()):
            if self.tabu_list[move] > 0:
                self.tabu_list[move] -= 1
            else:
                del self.tabu_list[move]

    # Main function to run the Tabu Search algorithm
    def run(self):
        current_solution = self.generate_initial_solution()
        current_makespan, _ = self.calculate_makespan(current_solution)
        best_solution = current_solution
        best_makespan = current_makespan
        iteration = 0
        evolution = []

        while iteration < self.max_iterations:
            neighbours = [self.swap_move(current_solution) for _ in range(10)]
            best_neighbour = None
            best_neighbour_makespan = float('inf')

            # Evaluate the neighbours
            for neighbour in neighbours:
                neighbour_makespan, _ = self.calculate_makespan(neighbour)

                # Aspiration criteria: if the neighbour is better, or it satisfies the aspiration
                if neighbour_makespan < best_neighbour_makespan or (self.aspiration_criteria and not self.is_tabu(neighbour)):
                    best_neighbour = neighbour
                    best_neighbour_makespan = neighbour_makespan

            # If the best neighbour is better than the current solution, accept it
            if best_neighbour_makespan < current_makespan:
                current_solution = best_neighbour
                current_makespan = best_neighbour_makespan
                # Add the move to the tabu list (convert to tuple)
                self.tabu_list[tuple(current_solution)] = self.tabu_tenure

            self.update_tabu_list()  # Update the tabu list for each iteration

            # Update best solution
            if current_makespan < best_makespan:
                best_solution = current_solution
                best_makespan = current_makespan

            evolution.append(best_makespan)
            iteration += 1

        best_makespan, best_schedule = self.calculate_makespan(best_solution)
        return best_solution, best_makespan, evolution, best_schedule

    # Plot the results (Evolution of Makespan and Gantt Chart)
    def plot_results(self, evolution, schedule):
        fig, axs = plt.subplots(1, 2, figsize=(18, 8))
        axs[0].plot(evolution, color='blue')
        axs[0].set_title('Evolution of Makespan', fontsize=14)
        axs[0].set_xlabel('Iteration', fontsize=12)
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
