class MultiAgent:
    def __init__(self, tf, grbm, vfs, env, num_sims):
        self.tf = tf
        self.rbm = grbm
        self.env = [grbm for _ in range(num_sims)]
        pass;

    def run_once(self):
        pass;
