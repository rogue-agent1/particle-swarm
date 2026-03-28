#!/usr/bin/env python3
"""particle_swarm - PSO optimizer."""
import argparse, random, math, sys

def pso(func, dim, bounds, n_particles=30, iterations=100, w=0.7, c1=1.5, c2=1.5):
    lo, hi = bounds
    particles = [[random.uniform(lo,hi) for _ in range(dim)] for _ in range(n_particles)]
    velocities = [[random.uniform(-1,1) for _ in range(dim)] for _ in range(n_particles)]
    p_best = [p[:] for p in particles]
    p_best_val = [func(p) for p in particles]
    g_best = min(range(n_particles), key=lambda i: p_best_val[i])
    g_best_pos = p_best[g_best][:]
    g_best_val = p_best_val[g_best]
    for it in range(iterations):
        for i in range(n_particles):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = (w*velocities[i][d] + c1*r1*(p_best[i][d]-particles[i][d]) + c2*r2*(g_best_pos[d]-particles[i][d]))
                particles[i][d] = max(lo, min(hi, particles[i][d] + velocities[i][d]))
            val = func(particles[i])
            if val < p_best_val[i]:
                p_best_val[i] = val; p_best[i] = particles[i][:]
                if val < g_best_val:
                    g_best_val = val; g_best_pos = particles[i][:]
        if it % (iterations//10) == 0:
            print(f"Iter {it:4d}: best={g_best_val:.6f}")
    return g_best_pos, g_best_val

BENCHMARKS = {
    "sphere": lambda x: sum(xi**2 for xi in x),
    "rastrigin": lambda x: 10*len(x) + sum(xi**2 - 10*math.cos(2*math.pi*xi) for xi in x),
    "rosenbrock": lambda x: sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1)),
    "ackley": lambda x: -20*math.exp(-0.2*math.sqrt(sum(xi**2 for xi in x)/len(x))) - math.exp(sum(math.cos(2*math.pi*xi) for xi in x)/len(x)) + 20 + math.e,
}

def main():
    p = argparse.ArgumentParser(description="Particle Swarm Optimization")
    p.add_argument("--func", choices=list(BENCHMARKS.keys()), default="sphere")
    p.add_argument("-d","--dim", type=int, default=10)
    p.add_argument("-n","--particles", type=int, default=30)
    p.add_argument("-i","--iters", type=int, default=200)
    a = p.parse_args()
    pos, val = pso(BENCHMARKS[a.func], a.dim, (-5.12, 5.12), a.particles, a.iters)
    print(f"\nOptimum: {val:.6f}")
    print(f"Position: [{', '.join(f'{x:.4f}' for x in pos[:5])}{'...' if len(pos)>5 else ''}]")

if __name__ == "__main__": main()
