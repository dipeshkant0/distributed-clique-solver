#!/usr/bin/env python3
"""
Solution verifier for Budgeted Maximum Weight Clique.
Validates:
1. All vertices in the clique are within valid range.
2. Every pair of vertices in the clique has an edge (Clique property).
3. Sum of costs <= Budget.
4. Sum of profits matches reported max profit.
"""

import sys

def verify_clique(graph_file, output_file):
    with open(graph_file, "r") as f:
        first_line = f.readline().strip().split()
        if not first_line:
            print("Empty graph file.")
            sys.exit(1)
        N = int(first_line[0])
        E = int(first_line[1])
        B = int(first_line[2])
        
        profits = []
        costs = []
        for _ in range(N):
            line = f.readline().strip().split()
            profits.append(int(line[0]))
            costs.append(int(line[1]))
            
        adj = set()
        for _ in range(E):
            line = f.readline().strip().split()
            u, v = int(line[0]), int(line[1])
            adj.add((u, v))
            adj.add((v, u))
            
    with open(output_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
        if not lines:
            print("Empty output file.")
            sys.exit(1)
        reported_profit = int(lines[0])
        clique = [int(x) for x in lines[1].split()] if len(lines) > 1 else []
        
    print(f"Reported Profit: {reported_profit}")
    print(f"Clique Vertices ({len(clique)}): {clique}")
    
    # 1. Range check
    for v in clique:
        assert 0 <= v < N, f"Vertex {v} out of range [0, {N})!"
        
    # 2. Clique check (complete subgraph)
    for i in range(len(clique)):
        for j in range(i + 1, len(clique)):
            u, v = clique[i], clique[j]
            assert (u, v) in adj, f"Vertices {u} and {v} are not connected by an edge!"
            
    # 3. Budget check
    total_cost = sum(costs[v] for v in clique)
    assert total_cost <= B, f"Total cost {total_cost} exceeds budget {B}!"
    
    # 4. Profit check
    total_profit = sum(profits[v] for v in clique)
    assert total_profit == reported_profit, f"Calculated profit {total_profit} does not match reported profit {reported_profit}!"
    
    print(f"[PASS] Solution is valid! Cost: {total_cost}/{B}, Profit: {total_profit}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_clique.py <graph_file> <output_file>")
        sys.exit(1)
    verify_clique(sys.argv[1], sys.argv[2])
