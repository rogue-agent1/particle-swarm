#!/usr/bin/env python3
"""particle_swarm - Particle Swarm Optimization."""
import argparse, random, math

def sphere(x): return sum(xi**2 for xi in x)
def rastrigin(x): return 10*len(x) + sum(xi**2 - 10*math.cos(2*math.pi*xi) for xi in x)
def rosenbrock(x): return sum(100*(x[i+1]-x[i]**2)**2 + (1-x[i])**2 for i in range(len(x)-1))

FUNCTIONS = {"sphere": (sphere, (-5,5)), "rastrigin": (rastrigin, (-5.12,5.12)), "rosenbrock": (rosenbrock, (-5,10))}

def pso(func, dims, bounds, n_particles=30, iterations=200, w=0.7, c1=1.5, c2=1.5):
    lo, hi = bounds
    particles = [[random.uniform(lo, hi) for _ in range(dims)] for _ in range(n_particles)]
    velocities = [[random.uniform(-1,1) for _ in range(dims)] for _ in range(n_particles)]
    pbest = [p[:] for p in particles]
    pbest_val = [func(p) for p in particles]
    gbest = min(pbest, key=func)[:]; gbest_val = func(gbest)
    for it in range(iterations):
        for i in range(n_particles):
            for d in range(dims):
                r1, r2 = random.random(), random.random()
                velocities[i][d] = w*velocities[i][d] + c1*r1*(pbest[i][d]-particles[i][d]) + c2*r2*(gbest[d]-particles[i][d])
                particles[i][d] += velocities[i][d]
                particles[i][d] = max(lo, min(hi, particles[i][d]))
            val = func(particles[i])
            if val < pbest_val[i]: pbest[i] = particles[i][:]; pbest_val[i] = val
            if val < gbest_val: gbest = particles[i][:]; gbest_val = val
        if it % (iterations//5) == 0:
            print(f"Iter {it:4d}: best={gbest_val:.6f}")
    return gbest, gbest_val

def main():
    p = argparse.ArgumentParser(description="Particle Swarm Optimization")
    p.add_argument("-f", "--function", choices=list(FUNCTIONS.keys()), default="sphere")
    p.add_argument("-d", "--dims", type=int, default=10)
    p.add_argument("-n", "--particles", type=int, default=30)
    p.add_argument("-i", "--iterations", type=int, default=200)
    args = p.parse_args()
    func, bounds = FUNCTIONS[args.function]
    best, val = pso(func, args.dims, bounds, args.particles, args.iterations)
    print(f"\nBest value: {val:.8f}")
    print(f"Solution: [{', '.join(f'{x:.4f}' for x in best[:5])}{'...' if len(best)>5 else ''}]")

if __name__ == "__main__":
    main()
