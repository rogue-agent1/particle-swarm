#!/usr/bin/env python3
"""particle_swarm - PSO optimizer."""
import sys, random, math
def pso(cost_fn, dims, n_particles=30, iterations=200, bounds=(-10,10)):
    lo, hi = bounds
    particles = [[random.uniform(lo,hi) for _ in range(dims)] for _ in range(n_particles)]
    velocities = [[random.uniform(-1,1) for _ in range(dims)] for _ in range(n_particles)]
    p_best = [list(p) for p in particles]
    p_best_cost = [cost_fn(p) for p in particles]
    g_best = min(p_best, key=cost_fn); g_best_cost = cost_fn(g_best)
    w, c1, c2 = 0.7, 1.5, 1.5
    for it in range(iterations):
        for i in range(n_particles):
            for d in range(dims):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = w*velocities[i][d] + c1*r1*(p_best[i][d]-particles[i][d]) + c2*r2*(g_best[d]-particles[i][d])
                particles[i][d] = max(lo, min(hi, particles[i][d]+velocities[i][d]))
            cost = cost_fn(particles[i])
            if cost < p_best_cost[i]: p_best[i]=list(particles[i]); p_best_cost[i]=cost
            if cost < g_best_cost: g_best=list(particles[i]); g_best_cost=cost
        if it % 50 == 0: print(f"Iter {it}: best={g_best_cost:.6f}")
    return g_best, g_best_cost
if __name__=="__main__":
    def rastrigin(x): return 10*len(x)+sum(xi**2-10*math.cos(2*math.pi*xi) for xi in x)
    dims = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    best, cost = pso(rastrigin, dims, bounds=(-5.12,5.12))
    print(f"Rastrigin({dims}D): min={cost:.6f}")
    print(f"At: [{', '.join(f'{x:.4f}' for x in best)}]")
