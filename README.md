# A Collection of Motion Planning Algorithms

This repository contains implementations for the algorithms discussed in [1]. These are:

 - Probabilistic RoadMaps (PRM, `prm`). 

 - Simplified Probabilistic RoadMaps (sPRM, `sprm`).

 - _k_-nearest Simplified Probabilistic RoadMaps (k-sPRM, `ksprm`).

 - Rapidly-exploring Random Trees (RRT, `rrt`).

 - Optimal Probabilistic RoadMaps (PRM*, `prm_star`).

 - _k_-nearest optimal Probabilistic RoadMaps (k-PRM*, `kprm_star`).

 - Rapidly-exploring Random Graph (RRG, `rrg`).

 - _k_-nearest Rapidly-exploring Random Graph (k-RRG, `krrg`).

 - Optimal Rapidly-exploring Random Trees (RRT*, `rrt_star`).

 - _k_-nearest optimal Rapidly-exploring Random Trees (k-RRT*, `krrt_star`).

To use the algorithms, add `src/motionplanning` to your path, the algorithms can be found in the `algorithms` subpackage (e.g. to import the PRM implementation one can use `from motionplanning.algorithms import prm`).

---

[1] KARAMAN, Sertac; FRAZZOLI, Emilio. Sampling-based algorithms for optimal motion planning. _The international journal of robotics research_, 2011, 30.7: 846-894.
