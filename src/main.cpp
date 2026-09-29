#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <bitset>
#include <mpi.h> 

using namespace std;

// Maximum number of nodes in our graphs
const int MAXN = 1205; 

// Graph details
int N, E, B; // Nodes, Edges, Budget
int profit[MAXN];
int cost[MAXN];

bitset<MAXN> adj[MAXN];
vector<int> memory_bank[MAXN]; 

// Global variables to track our best solution across all MPI processes
int global_bound = 0;       
int local_max_profit = 0;   
vector<int> best_clique;    

// Variables used for the Graph Coloring bound
bitset<MAXN> color_masks[MAXN]; 
int node_colors[MAXN];
bitset<MAXN> global_color_summed;

int efficiency_order[MAXN]; // Stores nodes sorted by profit/cost ratio
int eff_rank[MAXN];         // Stores the rank of each node for quick lookups

int my_rank, num_ranks;
int nodes_visited = 0; 
bool knapsack_first; // Flag to decide which bound to check first based on graph density

int get_graph_coloring_bound(const vector<int>& orig_candidates) {
    if (orig_candidates.empty()) return 0;
    
    int total_colors_used = 0;

    for (auto it = orig_candidates.rbegin(); it != orig_candidates.rend(); ++it) {
        int node = *it;
        int assigned_color = 0;
        
        // find first color that satisfies the condition that no adjacent node has the same color
        while (assigned_color < total_colors_used && (color_masks[assigned_color] & adj[node]).any()) {
            assigned_color++;
        }
        
        if (assigned_color == total_colors_used) {
            color_masks[assigned_color].reset(); 
            total_colors_used++;
        }
        
        color_masks[assigned_color].set(node);
        node_colors[node] = assigned_color;
    }

    // Sum up the highest profit from each color group used
    int max_possible_profit = 0;
    global_color_summed.reset(); 
    
    for (auto it = orig_candidates.rbegin(); it != orig_candidates.rend(); ++it) {
        int c = node_colors[*it];
        if (!global_color_summed.test(c)) {
            max_possible_profit += profit[*it];
            global_color_summed.set(c); // Mark this color as counted
        }
    }
    return max_possible_profit;
}

int get_fractional_knapsack_bound(const vector<int>& orig_candidates, int remaining_budget) {
    
    if (orig_candidates.empty() || remaining_budget <= 0) return 0;
    
    int max_p = 0;
    int used_b = 0;

    // threshold of 40 is chosen based on empirical testing to balance sorting overhead vs membership checking
    if (orig_candidates.size() < 40) {
        static int local_cands[MAXN];
        int sz = orig_candidates.size();
        for(int i = 0; i < sz; ++i) local_cands[i] = orig_candidates[i];

        // Sort based on the efficiency rank we calculated in main()
        sort(local_cands, local_cands + sz, [](int a, int b){
            return eff_rank[a] < eff_rank[b];
        });

        // Greedily pick items until the bag is full
        for(int i = 0; i < sz; ++i) {
            int node = local_cands[i];
            int rem = remaining_budget - used_b;
            if (rem <= 0) break;
            if (cost[node] <= rem) {
                max_p += profit[node];
                used_b += cost[node];
            } else {
                max_p += (int)((long long)profit[node] * rem / cost[node]);
                break;
            }
        }
        return max_p;
    }

    // bitset is faster for large candidate sets because we can quickly check membership
    static bitset<MAXN> membership;
    membership.reset();
    for (int v : orig_candidates) membership.set(v);

    for (int i = 0; i < N; ++i) {
        int node = efficiency_order[i];
        if (membership.test(node)) {
            int rem = remaining_budget - used_b;
            if (rem <= 0) break;
            if (cost[node] <= rem) {
                max_p += profit[node];
                used_b += cost[node];
            } else {
                max_p += (int)((long long)profit[node] * rem / cost[node]);
                break;
            }
        }
    }
    return max_p;
}

// The main recursive backtracking function that explores potential cliques.
void solve(int depth, vector<int>& candidates, int current_profit, int current_cost, vector<int>& current_clique) {
    
    if (candidates.empty()) return; // Base case: no more nodes to add

    // Periodically check for new global bounds from other processes to improve pruning
    if ((++nodes_visited & 8191) == 0) {
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        while (flag) {
            int received_max;
            MPI_Recv(&received_max, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (received_max > global_bound) global_bound = received_max; 
            MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
        }
    }

    if (knapsack_first) {
        if (current_profit + get_fractional_knapsack_bound(candidates, B - current_cost) <= global_bound) return;
        if (current_profit + get_graph_coloring_bound(candidates) <= global_bound) return;
    } else {
        if (current_profit + get_graph_coloring_bound(candidates) <= global_bound) return;
        if (current_profit + get_fractional_knapsack_bound(candidates, B - current_cost) <= global_bound) return;
    }

    // We process candidates backwards (highest profit first)
    while (!candidates.empty()) {
        int next_node = candidates.back();
        candidates.pop_back();

        int next_cost = current_cost + cost[next_node];
        
        // If adding this node doesn't exceed our budget
        if (next_cost <= B) {
            int new_profit = current_profit + profit[next_node];
            int new_size = current_clique.size() + 1;

            // Check if we found a new best clique
            if (new_profit > local_max_profit || (new_profit == local_max_profit && new_size > best_clique.size())) {
                local_max_profit = new_profit;
                best_clique = current_clique;
                best_clique.push_back(next_node);

                // If our new local best is better than the global best then tell everyone
                if (local_max_profit > global_bound) {
                    global_bound = local_max_profit;
                    for (int r = 0; r < num_ranks; r++) {
                        if (r != my_rank) MPI_Bsend(&global_bound, 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                    }
                }
            }

            // Prepare the candidates for the next level deep
            vector<int>& next_candidates = memory_bank[depth + 1];
            next_candidates.clear();
            
            int remaining_budget = B - next_cost;

            // Only pass down nodes that are connected to the node we just added
            for (int cand : candidates) {
                if (adj[next_node].test(cand) && cost[cand] <= remaining_budget) {
                    next_candidates.push_back(cand);
                }
            }

            current_clique.push_back(next_node);
            solve(depth + 1, next_candidates, new_profit, next_cost, current_clique);
            current_clique.pop_back();
        }
    }
}

int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (argc < 3) {
        if (my_rank == 0) cerr << "Usage: mpirun -np <ranks> ./main <input_file> <output_file>\n";
        MPI_Finalize();
        return 1;
    }
    
    // Attach a buffer so we can send messages asynchronously without blocking
    int buffer_size = 100 * num_ranks * (sizeof(int) + MPI_BSEND_OVERHEAD);
    void* mpi_buffer = malloc(buffer_size);
    MPI_Buffer_attach(mpi_buffer, buffer_size);

    vector<int> flat_edges;
    
    // Only Rank 0 reads the file to avoid file I/O bottlenecks
    if (my_rank == 0) {
        ifstream fin(argv[1]);
        if (!fin) MPI_Abort(MPI_COMM_WORLD, 1);
        fin >> N >> E >> B;
        for (int i = 0; i < N; i++) fin >> profit[i] >> cost[i];
        
        flat_edges.resize(E * 2);
        for (int i = 0; i < E * 2; i += 2) {
            fin >> flat_edges[i] >> flat_edges[i + 1];
        }
        fin.close();
    }

    // Rank 0 broadcasts the basic stats to all other processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&E, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (my_rank != 0) flat_edges.resize(E * 2);

    // Broadcast the heavy arrays
    MPI_Bcast(profit, N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cost, N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_edges.data(), E * 2, MPI_INT, 0, MPI_COMM_WORLD);

    // Build the adjacency matrix using bitsets
    for (int i = 0; i < E * 2; i += 2) {
        int u = flat_edges[i];
        int v = flat_edges[i + 1];
        adj[u].set(v); 
        adj[v].set(u);
    }

    //Sort all nodes by efficiency (profit/cost ratio)
    for (int i = 0; i < N; ++i) efficiency_order[i] = i;
    sort(efficiency_order, efficiency_order + N, [](int a, int b) {
        long long lhs = (long long)profit[a] * cost[b];
        long long rhs = (long long)profit[b] * cost[a];
        if (lhs != rhs) return lhs > rhs;
        if (profit[a] != profit[b]) return profit[a] > profit[b];
        return a < b; 
    });

    // Create a reverse-lookup for the efficiency order
    for (int i = 0; i < N; ++i) {
        eff_rank[efficiency_order[i]] = i;
    }

    knapsack_first = (E > (N * N) / 3);

    vector<int> initial_candidates(N);
    iota(initial_candidates.begin(), initial_candidates.end(), 0);
    
    sort(initial_candidates.begin(), initial_candidates.end(), [](int a, int b) { 
        if (profit[a] != profit[b]) return profit[a] < profit[b];
        return a > b; 
    });

    vector<int> current_clique;
    int task_id = 0;


    // Each MPI process takes turns picking the next node to explore from the initial candidates list.
    while (!initial_candidates.empty()) {
        int next_node = initial_candidates.back();
        initial_candidates.pop_back();

        if (task_id % num_ranks == my_rank) {

            if (cost[next_node] <= B) {
                if (profit[next_node] > local_max_profit) {
                    local_max_profit = profit[next_node];
                    best_clique = {next_node};
                    if (local_max_profit > global_bound) global_bound = local_max_profit;
                }

                vector<int>& next_candidates = memory_bank[1];
                next_candidates.clear();
                
                int remaining_budget = B - cost[next_node];
                for (int cand : initial_candidates) {
                    if (adj[next_node].test(cand) && cost[cand] <= remaining_budget) {
                        next_candidates.push_back(cand);
                    }
                }

                current_clique.push_back(next_node);
                solve(1, next_candidates, profit[next_node], cost[next_node], current_clique);
                current_clique.pop_back();
            }
        }
        task_id++;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Flush any remaining messages out of the queue so we don't crash
    int flag;
    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    while (flag) {
        int dump;
        MPI_Recv(&dump, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }

    // Compare all the local bests to find the ultimate global winner
    struct { int val; int rank; } local_res, global_res;
    local_res.val = local_max_profit;
    local_res.rank = my_rank;
    MPI_Allreduce(&local_res, &global_res, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);

    int best_rank = global_res.rank;

    // Send the winning clique to Rank 0
    if (my_rank == best_rank) {
        if (my_rank != 0) {
            int size = best_clique.size();
            MPI_Send(&size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(best_clique.data(), size, MPI_INT, 0, 2, MPI_COMM_WORLD);
        }
    } 
    else if (my_rank == 0) {
        int size;
        MPI_Recv(&size, 1, MPI_INT, best_rank, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        best_clique.resize(size);
        MPI_Recv(best_clique.data(), size, MPI_INT, best_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        local_max_profit = global_res.val; 
    }

    // double end_time = MPI_Wtime();

    // Rank 0 writes the final answer to the output file
    if (my_rank == 0) {
        sort(best_clique.begin(), best_clique.end());
        
        ofstream fout(argv[2]);
        fout << local_max_profit << "\n";
        for (size_t i = 0; i < best_clique.size(); i++) {
            fout << best_clique[i] << (i == best_clique.size() - 1 ? "" : " ");
        }
        fout << "\n";
        fout.close();
    }

    // Clean up the MPI buffer
    int dummy_size;
    void* dummy_buf;
    MPI_Buffer_detach(&dummy_buf, &dummy_size);
    free(mpi_buffer);

    MPI_Finalize();
    return 0;
}