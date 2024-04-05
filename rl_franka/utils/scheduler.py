class LinearScheduler:
    def __init__(self, start_value, end_value, num_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.current_step = 0

    def step(self):
        t = min(self.current_step / self.num_steps, 1.0)
        value = self.start_value + (self.end_value - self.start_value) * t
        self.current_step += 1
        return value