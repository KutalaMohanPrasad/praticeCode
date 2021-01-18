import numpy as np

class Particle:
    """
    Particle class represents a solution inside a pool(Swarm).
    """
    def __init__(self, no_dim, x_range, v_range):
        """
        Particle class constructor
        :param no_dim: int
            No of dimensions.
        :param x_range: tuple(double)
            Min and Max value(range) of dimension.
        :param v_range: tuple(double)
            Min and Max value(range) of velocity.
        """
        self.x = np.random.uniform(x_range[0], x_range[1], (no_dim, )) #particle position in each dimension...
        self.v = np.random.uniform(v_range[0], v_range[1], (no_dim, )) #particle velocity in each dimension...
        self.pbest = np.inf
        self.pbestpos = np.zeros((no_dim, ))

class PSwarm:


    def __init__(self, no_particle, no_dim, x_range, v_range, iw_range, c):

        self.p = np.array([Particle(no_dim, x_range, v_range) for i in range(no_particle)])
        self.gbest = np.inf
        self.gbestpos = np.zeros((no_dim,))
        self.x_range = x_range
        self.v_range = v_range
        self.iw_range = iw_range
        self.c0 = c[0]
        self.c1 = c[1]
        self.no_dim = no_dim


    def optimize(self, iter,tot_train_rmse):
        total_t=0
        for i in range(iter):
            for particle in self.p:
                fitness = tot_train_rmse

                if (fitness < particle.pbest).any():
                    particle.pbest = fitness
                    particle.pbestpos = particle.x.copy()

                if (fitness < self.gbest).any():
                    self.gbest = fitness
                    self.gbestpos = particle.x.copy()

            for particle in self.p:
                # Here iw is inertia weight...
                iw = np.random.uniform(self.iw_range[0], self.iw_range[1], 1)[0]
                particle.v = iw * particle.v + (self.c0 * np.random.uniform(0.0, 1.0, (self.no_dim,)) * \
                                                (particle.pbestpos - particle.x)) + (
                                         self.c1 * np.random.uniform(0.0, 1.0, (self.no_dim,)) \
                                         * (self.gbestpos - particle.x))
                #self.gbest = fitness*self.gbestpos
                total_t +=self.gbestpos[i]
                # particle.v = particle.v.clip(min=self.v_range[0], max=self.v_range[1])
                particle.x = particle.x + particle.v
                # particle.x = particle.x.clip(min=self.x_range[0], max=self.x_range[1])

            # if i % print_step == 0:
            #     print('iteration#: ', i + 1, ' loss: ', fitness)

        print("global best using PSO: ", self.gbestpos[i])
        return total_t


    def get_best_solution(self):

        return self.gbestpos


