import argparse, random, math

def sphere(x): return sum(v**2 for v in x)
def rastrigin(x): return 10*len(x) + sum(v**2 - 10*math.cos(2*math.pi*v) for v in x)

FUNCS = {"sphere": sphere, "rastrigin": rastrigin}

def pso(func, dim=5, n_particles=30, iters=200, seed=None):
    if seed: random.seed(seed)
    w, c1, c2 = 0.7, 1.5, 1.5
    pos = [[random.uniform(-5, 5) for _ in range(dim)] for _ in range(n_particles)]
    vel = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(n_particles)]
    pbest = [p[:] for p in pos]
    pbest_f = [func(p) for p in pos]
    gi = min(range(n_particles), key=lambda i: pbest_f[i])
    gbest, gbest_f = pbest[gi][:], pbest_f[gi]
    for it in range(iters):
        for i in range(n_particles):
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                vel[i][d] = w*vel[i][d] + c1*r1*(pbest[i][d]-pos[i][d]) + c2*r2*(gbest[d]-pos[i][d])
                pos[i][d] += vel[i][d]
            f = func(pos[i])
            if f < pbest_f[i]: pbest[i], pbest_f[i] = pos[i][:], f
            if f < gbest_f: gbest, gbest_f = pos[i][:], f
        if it % 40 == 0: print(f"Iter {it:4d}: best={gbest_f:.6f}")
    print(f"Final: best={gbest_f:.6f}")

def main():
    p = argparse.ArgumentParser(description="PSO optimizer")
    p.add_argument("func", choices=FUNCS.keys())
    p.add_argument("-d", "--dim", type=int, default=5)
    p.add_argument("-n", "--particles", type=int, default=30)
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    pso(FUNCS[args.func], args.dim, args.particles, seed=args.seed)

if __name__ == "__main__":
    main()
