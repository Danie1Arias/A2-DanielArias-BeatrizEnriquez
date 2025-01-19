import random

class JobShop:
    def __init__(self, jobs, machines= 3):
        self.jobs = jobs
        # Determinar el número de máquinas si no se proporciona explícitamente
        self.machines = machines if machines is not None else len(jobs[0])
        self.num_jobs = len(jobs)
        print("NUM_MACHINES:", self.machines)
        print("NUM_JOBS:", len(jobs))

    def makespan(self, schedule):
        machine_times = [0] * self.machines
        job_times = [0] * self.num_jobs

        for job, machine in schedule:
            processing_time = self.jobs[job][machine]
            start_time = max(machine_times[machine], job_times[job])
            end_time = start_time + processing_time
            machine_times[machine] = end_time
            job_times[job] = end_time

        return max(machine_times)


def generate_initial_solution(problem):
    schedule = []
    for job_id in range(problem.num_jobs):
        for machine_id in range(problem.machines):
            schedule.append((job_id, machine_id))
    return schedule


def get_neighbors(solution):
    neighbors = []
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if solution[i][0] == solution[j][0]:  # Solo intercambiar si son del mismo trabajo
                neighbor = solution[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                neighbors.append(neighbor)
    return neighbors


def tabu_search(problem, max_iter=1000, tabu_size=100):
    current_solution = generate_initial_solution(problem)
    best_solution = current_solution
    best_makespan = problem.makespan(best_solution)

    tabu_list = []
    iterations = 0

    while iterations < max_iter:
        neighbors = get_neighbors(current_solution)
        best_neighbor = None
        best_neighbor_makespan = float('inf')

        for neighbor in neighbors:
            if neighbor not in tabu_list:
                neighbor_makespan = problem.makespan(neighbor)
                if neighbor_makespan < best_neighbor_makespan:
                    best_neighbor = neighbor
                    best_neighbor_makespan = neighbor_makespan

        current_solution = best_neighbor
        current_makespan = best_neighbor_makespan

        if current_makespan < best_makespan:
            best_solution = current_solution
            best_makespan = current_makespan

        tabu_list.append(current_solution)
        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

        iterations += 1

    return best_solution, best_makespan


data1  = [
    [0, 3, 1, 2, 2, 2],
    [0, 2, 2, 1, 1, 4],
    [1, 4, 2, 3]
]
data2 = [
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


num_machines = 3  # Number of machines

problem = JobShop(data1, num_machines)

best_solution, best_makespan = tabu_search(problem)
print("Mejor solución encontrada:", best_solution)
print("Makespan:", best_makespan)
