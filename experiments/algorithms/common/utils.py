def linear_schedule(start_epsilon: float, end_epsilon: float, duration: int, timestep: int):
    slope = (end_epsilon - start_epsilon) / duration
    return max(slope * timestep + start_epsilon, end_epsilon)