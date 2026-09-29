MPICXX = mpicxx
CXXFLAGS = -std=c++17 -O3

TARGET = mpi_clique
SRC = src/main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(MPICXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

test: $(TARGET)
	@echo "Generating sample graph..."
	python3 scripts/generate_graph.py 80 0.3 300 sample_graph.txt
	@echo "Running MPI solver on 4 ranks..."
	mpirun -np 4 ./$(TARGET) sample_graph.txt output.txt
	@echo "Verifying solution..."
	python3 scripts/verify_clique.py sample_graph.txt output.txt
	@rm -f sample_graph.txt output.txt

clean:
	rm -f $(TARGET) sample_graph.txt output.txt

.PHONY: all test clean
