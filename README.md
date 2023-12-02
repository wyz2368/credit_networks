## Empirical Game-Theoretic Analysis for Credit Networks
This project applies empirical game-theoretic analysis (EGTA) for credit networks. 
Currently, the code adheres to the prepayment operation. 

### Code Structure
#### *classic_EGTA* 
1. **nash_solvers**: This folder contains replicator dynamics and gambit for analyzing games.
2. **clearing.py**: Functions for clearing a credit network.
3. **egta_solver.py**: The main solver class for EGTA, including simulating profiles and equilibrium computation.
4. **evaluation.py**: Functions for analyzing the equilibria and computing price of anarchy. 
5. **net_generator.py**: Functions for generating instances of credit networks.
6. **player_reduction.py**: Functions for player reduction.
7. **strategies**: Definitions for heuristic strategies.
8. **symmetric_utils.py**: Functions for creating compact representation for symmetric games, 
converting a compact representation to complete payoff matrix, and computing pure-strategy equilibria.

#### *envs*
1. **prepayment_net**: A Pettingzoo environment for credit networks with prepayment.

#### *instances*
This folder includes instances generated with different network parameters.

#### *example.py*
An example for running EGTA for credit networks.

