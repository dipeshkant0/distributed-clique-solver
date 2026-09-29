#!/usr/bin/env python3
"""
Random Graph Generator for Budgeted Maximum Weight Clique Problem.
Format:
N E B
profit_0 cost_0
profit_1 cost_1
...
u_0 v_0
u_1 v_1
...
"""

import sys
import random

def generate_graph(n=100, density=0.25, budget=500, max_profit=100, max_cost=50, output_file="sample_graph.txt", seed=42):
    random.seed(seed)
    
    edges = set()
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < density:
                edges.add((u, v))
                
    edges_list = list(edges)
    e = len(edges_list)
    
    profits = [random.randint(10, max_profit) for _ in range(n)]
    costs = [random.randint(5, max_cost) for _ in range(n)]
    
    with open(output_file, "w") as f:
        f.write(f"{n} {e} {budget}\n")
        for i in range(n):
            f.write(f"{profits[i]} {costs[i]}\n")
        for u, v in edges_list:
            f.write(f"{u} {v}\n")
            
    print(f"Generated graph: {n} vertices, {e} edges, budget {budget} -> {output_file}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    density = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    out = sys.argv[4] if len(sys.argv) > 4 else "sample_graph.txt"
    generate_graph(n, density, budget, output_file=out)
