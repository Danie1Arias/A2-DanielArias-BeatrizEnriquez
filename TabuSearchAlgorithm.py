import random
from Utils import parse_data, generate_initial_solution, calculate_makespan

class TabuSearch:
    def __init__(self, data, max_iterations=1000, tabu_tenure=10, aspiration_criteria=True):
        self.data = data
        self.jobs = parse_data(data)
        self.num_jobs = len(self.jobs)
        self.num_tasks_per_job = len(self.jobs[0])
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.tabu_list = {}
        self.best_solution = None
        self.best_makespan = float('inf')
        self.aspiration_criteria = aspiration_criteria

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
        current_solution = generate_initial_solution(self.jobs)
        current_makespan, _ = calculate_makespan(current_solution, self.jobs)
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
                neighbour_makespan, _ = calculate_makespan(neighbour, self.jobs)

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

        best_makespan, best_schedule = calculate_makespan(best_solution, self.jobs)
        return best_solution, best_makespan, evolution, best_schedule
