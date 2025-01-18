import random
import numpy as np

# Define the structure of the Job Shop Scheduling Problem (JSSP)
class JobShop:
    def __init__(self, jobs, machines):
        self.jobs = jobs        # List of jobs, where each job is a list of processing times for each machine
        self.machines = machines  # Number of machines
        self.num_jobs = len(jobs)  # Number of jobs

    def makespan(self, schedule):
        """
        Calculates the makespan (total execution time) given a task schedule.
        """
        machine_times = [0] * self.machines  # Current processing times for each machine
        job_end_times = [0] * self.num_jobs  # Completion times for each job

        for job, machine in schedule:
            # Calculate the start time of the task
            start_time = max(machine_times[machine], job_end_times[job])  
            # Calculate the end time of the task
            end_time = start_time + self.jobs[job][machine]  
            # Update the machine's processing time
            machine_times[machine] = end_time  
            # Update the job's completion time
            job_end_times[job] = end_time  

        # The makespan is the time when the last job finishes on any machine
        return max(machine_times)


# Function to generate a random initial solution
def generate_initial_solution(problem):
    schedule = []
    for job_id in range(problem.num_jobs):
        for machine_id in range(problem.machines):
            schedule.append((job_id, machine_id))  # Pair jobs with machines
    random.shuffle(schedule)  # Shuffle tasks randomly to create an initial solution
    return schedule

# Function to obtain neighbors of a solution
def get_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            # Swap two tasks to generate a neighbor solution
            neighbor = solution[:]
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]  
            neighbors.append(neighbor)
    return neighbors

# Tabu search algorithm
def tabu_search(problem, max_iter=1000, tabu_size=50):
    current_solution = generate_initial_solution(problem)
    best_solution = current_solution
    best_makespan = problem.makespan(best_solution)

    tabu_list = []  # List of tabu solutions
    iterations = 0

    while iterations < max_iter:
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_makespan = float('inf')

        for neighbor in neighbors:
            # Check if the neighbor is not in the tabu list
            if neighbor not in tabu_list:
                neighbor_makespan = problem.makespan(neighbor)
                # Update the best neighbor if it has a lower makespan
                if neighbor_makespan < best_neighbor_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = neighbor_makespan

        # Update the current solution
        current_solution = best_neighbor
        current_makespan = best_neighbor_makespan

        # Update the best solution found so far
        if current_makespan < best_makespan:
            best_solution = current_solution
            best_makespan = current_makespan

        # Add the current solution to the tabu list
        tabu_list.append(current_solution)
        # Remove the oldest solution if the tabu list exceeds its size limit
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        iterations += 1

    return best_solution, best_makespan


# Example usage

# Define the JSSP: processing times for jobs on machines
# Each row represents a job, and each column represents a machine
jobs = [
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

machines = 3  # Number of machines

# Create the problem instance
problem = JobShop(jobs, machines)

# Execute the tabu search
best_solution, best_makespan = tabu_search(problem)

# Print the best solution found and its makespan
print("Best solution found (in terms of job and machine order):")
print(best_solution)
print(f"Makespan of the best solution: {best_makespan}")