# Research Sources

Updated: 2026-04-20 19:45 EDT

Access date for all sources: 2026-04-20.

## Collision Avoidance / ORCA / RVO2

### Optimal Reciprocal Collision Avoidance project page

- Authors/org: Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha; UNC.
- Link: https://gamma-web.iacs.umd.edu/ORCA/
- Direct source claims:
  - ORCA is a formal reciprocal collision avoidance approach for independent agents without communication.
  - Each pair of agents takes half the responsibility for avoidance.
  - Selecting each action reduces to a low-dimensional linear program.
  - Demonstrated in dense 2D/3D scenarios with thousands of agents in a few milliseconds.
- Repo implications/inferences:
  - Good near-term shield/baseline for `ContinuousMAPFEnv` because repo actions are velocity commands.
  - ORCA is local/reactive; do not expect it to solve global obstacle routing or MAPF deadlocks by itself.
  - Learned policy should provide preferred velocities or short-horizon waypoints; ORCA should be the safety projector.

### RVO2 Library documentation

- Authors/org: Jur van den Berg, Stephen J. Guy, Jamie Snape, Ming C. Lin, Dinesh Manocha; UNC/GAMMA.
- Links:
  - https://gamma-web.iacs.umd.edu/RVO2/documentation/2.0/
  - https://gamma-web.iacs.umd.edu/RVO2/documentation/2.0/using.html
- Direct source claims:
  - RVO2 is an implementation of ORCA for multi-agent simulation.
  - It can automatically use multiple processors.
  - User specifies agents, obstacles, and preferred velocities; simulator computes actual velocities that follow preferences while guaranteeing collision avoidance.
  - Obstacles must be specified before simulation and processed; goal/roadmap guidance is not part of RVO2 and must be supplied externally.
  - Simple goal-directed preferred velocity may not give convincing global navigation around obstacles; roadmaps/global planning may be needed.
- Repo implications/inferences:
  - Current repo's SDF-preconstraint + rvo2-agent-agent path is not full standard RVO2 obstacle handling; document this clearly.
  - Transformer policy should learn global/local intent; ORCA/RVO2 should not be treated as planner.
  - Obstacle-map benchmark failures should be investigated as guidance failures, not only shield failures.

### Official RVO2 GitHub repository

- Authors/org: Jamie Snape et al.; UNC/GAMMA.
- Link: https://github.com/snape/RVO2
- Direct source claims:
  - RVO2 is an open-source C++98 implementation of ORCA in 2D.
  - API specifies static obstacles, agents, and preferred velocities; step-by-step simulation; OpenMP parallelization.
  - Apache-2.0 license in current GitHub repo.
- Repo implications/inferences:
  - Prefer verifying whether Python `rvo2` binding in the experiment environment matches official semantics.
  - If `rvo2` is absent, repo `shield_type=orca` can silently become heuristic; benchmark tables must log true-vs-fallback shield implementation.

### Reciprocal n-Body Collision Avoidance

- Authors/org: Jur van den Berg, Stephen J. Guy, Ming Lin, Dinesh Manocha.
- Link: https://link.springer.com/chapter/10.1007/978-3-642-19457-3_1
- Direct source claims:
  - Derives sufficient conditions for collision-free motion from velocity obstacles.
  - Uses low-dimensional LPs.
  - Reports collision-free actions for thousands of robots in complex/dense scenarios in milliseconds.
- Repo implications/inferences:
  - Supports ORCA as a scalability baseline and shield.
  - Does not remove need for learned/global guidance because ORCA's guarantee is local conditional collision avoidance under modeling assumptions.

### Generalized reciprocal collision avoidance

- Authors/org: Daman Bareiss, Jur van den Berg.
- Link: https://journals.sagepub.com/doi/10.1177/0278364915576234
- Direct source claims:
  - Unifies velocity obstacles, acceleration-velocity obstacles, continuous control obstacles, and LQR-obstacles under control obstacles.
  - Extends reciprocal avoidance to heterogeneous nonlinear systems.
- Repo implications/inferences:
  - Keep architecture compatible with future acceleration/nonholonomic dynamics by representing state/action abstractions cleanly.
  - Do not hard-code single-integrator velocity outputs too deeply if long-term goal includes richer robot dynamics.

### Safety Barrier Certificates for Collision-Free Multirobot Systems

- Authors/org: Li Wang, Aaron D. Ames, Magnus Egerstedt.
- Link: https://authors.library.caltech.edu/records/tshzw-g4v69
- Direct source claims:
  - Safety barrier certificates modify nominal controllers to satisfy safety constraints.
  - Safety controller minimizes deviation from nominal control subject to constraints using a QP.
  - Starts centralized and develops decentralized nearby-robot variants.
- Repo implications/inferences:
  - CBF/PICBF is the principled long-term shield direction because it directly projects nominal learned controls under constraints.
  - ORCA can remain near-term because it is cheaper and already present; design learned-policy interface as nominal-control-in/safe-control-out to allow swapping in CBF later.

### Neural Graph Control Barrier Functions Guided Distributed Collision-Avoidance Multi-Agent Control

- Authors/org: Songyuan Zhang, Kunal Garg, Chuchu Fan; CoRL 2023/PMLR.
- Link: https://proceedings.mlr.press/v229/zhang23h.html
- Direct source claims:
  - Introduces graph CBFs for distributed collision avoidance with local information.
  - Learns neural GCBF certificate and distributed control with graph neural networks.
  - Addresses scalability/generalization and includes LiDAR point-cloud observations.
- Repo implications/inferences:
  - PICBF/GCBF is a viable secondary research track for shield-aware transformer policies.
  - Transformer local attention can be aligned with graph-CBF locality/communication radius.

## MAPF / Learned Local Policies / Lifelong MAPF

### Multi-Agent Pathfinding: Definitions, Variants, and Benchmarks

- Authors/org: Roni Stern, Nathan R. Sturtevant, Ariel Felner, Sven Koenig, Hang Ma, Thayne Walker, Jiaoyang Li, Dor Atzmon, Liron Cohen, T. K. Satish Kumar, Eli Boyarski, Roman Bartak.
- Links:
  - https://cris.biu.ac.il/en/publications/multi-agent-pathfinding-definitions-variants-and-benchmarks/
  - https://www.movingai.com/benchmarks/mapf.html
- Direct source claims:
  - MAPF is planning paths for multiple agents such that they can follow them concurrently without collisions.
  - The paper emphasizes unifying terminology for assumptions/objectives and provides grid benchmarks.
  - MovingAI MAPF benchmark collects maps/problems for broad comparison.
- Repo implications/inferences:
  - Final plan should explicitly define continuous-space assumptions: agent geometry, dynamics, time, collision metric, objectives.
  - Keep grid benchmark comparison only as historical baseline; continuous claims need their own protocol.

### Work Smarter Not Harder: Simple Imitation Learning with CS-PIBT Outperforms Large-Scale Imitation Learning for MAPF

- Authors/org: Rishi Veerapaneni, Arthur Jakobsson, Kevin Ren, Samuel Kim, Jiaoyang Li, Maxim Likhachev; ICRA 2025.
- Links:
  - https://arthurjakobsson.github.io/ssil_mapf/
  - https://colab.ws/articles/10.1109%2Ficra55743.2025.11128836
  - https://arcs-group.github.io/mapf/
- Direct source claims:
  - Large-scale imitation alone did not produce impressive results with their architecture.
  - Adding CS-PIBT one-step collision shielding dramatically improved learned local MAPF policies.
  - Future learned policies should include smart one-step collision shields and compare against shielded greedy baselines.
  - Longer-horizon/more complex planning is a key remaining target because one-step collisions can be efficiently resolved.
- Repo implications/inferences:
  - Directly supports keeping ORCA/PICBF shielding in continuous MAPF rather than asking transformer to learn safety alone.
  - Transformer should focus on intent/global coordination/deadlock avoidance, not merely one-step collision resolution.
  - Evaluation must include ORCA-only/shielded greedy baselines and shield intervention metrics.

### Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding

- Authors/org: Keisuke Okumura, Manao Machida, Xavier Defago, Yasumasa Tamura.
- Links:
  - https://www.ijcai.org/Proceedings/2019/76
  - https://www.sciencedirect.com/science/article/pii/S0004370222000923
- Direct source claims:
  - PIBT solves iterative MAPF using priorities, priority inheritance, and backtracking.
  - It is scalable, local/decentralizable, and guarantees reachability under graph conditions such as adjacent nodes on simple cycles.
- Repo implications/inferences:
  - Current `epibt` shield conceptually fits repo heritage but its continuous implementation should be treated as heuristic until tested.
  - Priority features could become transformer inputs for compatibility with PO-ORCA/EPIBT-style shields.

### Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding

- Authors/org: Keisuke Okumura; IJCAI 2023.
- Link: https://www.ijcai.org/proceedings/2023/28
- Direct source claims:
  - LaCAM uses lazy successor generation for scalable MAPF.
  - LaCAM* is anytime and eventually converges to optima under transition-cost assumptions.
  - Solved 99% of MAPF benchmark instances up to 1000 agents within 10 seconds in reported experiments.
- Repo implications/inferences:
  - LaCAM-derived discrete paths can be a better global-guidance expert than ORCA-only in obstacle maps.
  - Continuous data generation already has LaCAM3 hooks; roadmap should elevate this as an expert source for hard maps.

### Deploying Ten Thousand Robots: Scalable Imitation Learning for Lifelong MAPF

- Authors/org: He Jiang, Yutong Wang, Rishi Veerapaneni, Tanishq Duhan, Guillaume Sartoretti, Jiaoyang Li; ICRA 2025.
- Links:
  - https://www.researchgate.net/publication/385353545_Deploying_Ten_Thousand_Robots_Scalable_Imitation_Learning_for_Lifelong_Multi-Agent_Path_Finding
  - https://dblp.org/rec/conf/icra/JiangWVDSL25
- Direct source claims:
  - SILLM scales imitation learning for lifelong MAPF to 10,000 agents.
  - Combines learned local observations, communication module, collision resolution, and global guidance.
  - Reports outperforming learning/search baselines on large maps and real/virtual robot validation.
- Repo implications/inferences:
  - Reinforces architectural triad for this repo: learned policy + explicit shield + global guidance features.
  - Sparse/local communication is more appropriate than full global attention for high agent counts.

### Multi-Agent Pathfinding with Continuous Time

- Authors/org: Andreychuk et al. / Artificial Intelligence 2022.
- Link: https://www.sciencedirect.com/science/article/pii/S0004370222000029
- Direct source claims:
  - Standard MAPF often assumes discrete time and unit-duration actions.
  - Continuous-time MAPF addresses nontrivial issues around action durations and real-world applicability.
- Repo implications/inferences:
  - Continuous-space roadmap must define `dt`, integration semantics, and collision checking between timesteps, not only discrete sample positions.

### Multi-Agent Path Finding for Large Agents

- Authors/org: Jiaoyang Li, Pavel Surynek, Ariel Felner, Hang Ma, T. K. Satish Kumar, Sven Koenig; AAAI 2019.
- Link: https://publications.ri.cmu.edu/multi-agent-path-finding-for-large-agents
- Direct source claims:
  - Standard MAPF often assumes point/single-cell agents.
  - Large/geometric agents require adapting collision constraints.
- Repo implications/inferences:
  - `agent_radius` and circle-vs-obstacle checks are central, not implementation details.
  - Evaluation should vary radius/clearance and not claim equivalence to point-agent grid MAPF.

### Multi-Agent Path Finding in Continuous Spaces with Projected Diffusion Models

- Authors/org: Jinhao Liang, Jacob K. Christopher, Sven Koenig, Ferdinando Fioretto; AAAI-25 MAPF Workshop.
- Links:
  - https://womapf.github.io/aaai-25/pdf/Submission_38.pdf
  - https://idm-lab.org/bib/abstracts/Koen25d.html
- Direct source claims:
  - Continuous MAPF is hard because optimization struggles with scalability and discretization can be impractical.
  - Direct diffusion extension has feasibility problems for collision constraints.
  - Projected diffusion combines generative modeling with constrained optimization to produce feasible trajectories.
- Repo implications/inferences:
  - Flow matching for velocity chunks should be coupled to projection/shielding, not unconstrained generation.
  - Longer horizon trajectory/chunk generation is a promising later phase after one-step transformer policy is stable.

## Transformer / Attention / Multi-Agent Architecture

### Attention Is All You Need

- Authors/org: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan Gomez, Lukasz Kaiser, Illia Polosukhin.
- Link: https://huggingface.co/papers/1706.03762
- Direct source claims:
  - Transformer uses attention without recurrence/convolution and is more parallelizable for sequence modeling.
- Repo implications/inferences:
  - Transformer is attractive for batching variable agent sets and modeling interactions, but naive attention is quadratic.

### Set Transformer

- Authors/org: Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, Yee Whye Teh; ICML 2019.
- Link: https://proceedings.mlr.press/v97/lee19d
- Direct source claims:
  - Designed for set-structured data where outputs should not depend on element order.
  - Uses attention to model interactions among set elements.
  - Inducing-point attention reduces self-attention complexity from quadratic to linear in the number of elements.
- Repo implications/inferences:
  - Agent tokens are naturally a set; architecture must preserve permutation equivariance/invariance.
  - Inducing tokens are an option for global summaries if local sparse attention is insufficient.

### Self-Attention with Relative Position Representations

- Authors/org: Peter Shaw, Jakob Uszkoreit, Ashish Vaswani; NAACL 2018 / Google Research.
- Links:
  - https://research.google/pubs/pub46989
  - https://aclanthology.org/N18-2074/
- Direct source claims:
  - Relative position representations in self-attention improve over absolute position-only representations.
  - The approach generalizes to arbitrary graph-labeled inputs.
- Repo implications/inferences:
  - Current `RelativePosEncoding(edge_attr -> per-head bias)` is directionally appropriate.
  - Upgrade should encode relative position, distance, bearing, relative velocity, radius/clearance, and obstacle visibility as attention biases/features.

### Graphormer / Do Transformers Really Perform Badly for Graph Representation?

- Authors/org: Chengxuan Ying et al.; Microsoft Research / NeurIPS 2021.
- Link: https://www.microsoft.com/en-us/research/publication/do-transformers-really-perform-badly-for-graph-representation/
- Direct source claims:
  - Graphormer uses structural encodings to make Transformer effective on graph representation learning.
  - Structural information is necessary/effective for graph transformers.
  - The paper characterizes Graphormer expressivity and shows many GNN variants as special cases.
- Repo implications/inferences:
  - Transformer should not ignore graph structure; local neighbor graph, distances, obstacle-aware paths, and shield communication radius should be encoded explicitly.

### Multi-Agent Reinforcement Learning is a Sequence Modeling Problem / Multi-Agent Transformer

- Authors/org: Muning Wen, Jakub Kuba, Runji Lin, Weinan Zhang, Ying Wen, Jun Wang, Yaodong Yang; NeurIPS 2022.
- Links:
  - https://papers.nips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html
  - https://github.com/PKU-MARL/Multi-Agent-Transformer
- Direct source claims:
  - Casts cooperative MARL as sequence modeling mapping agent observation sequences to action sequences.
  - Official implementation uses encoder-decoder Transformer architecture.
- Repo implications/inferences:
  - For centralized training, ordering can be a training artifact, but execution should remain permutation-equivariant/local unless using autoregressive priority order intentionally.
  - Sequence/chunk outputs are plausible but should not introduce arbitrary agent-order dependence.

### AgentFormer

- Authors/org: Ye Yuan, Xinshuo Weng, Yanglan Ou, Kris Kitani; ICCV 2021 / CMU.
- Link: https://www.ri.cmu.edu/publications/agentformer-agent-aware-transformers-for-socio-temporal-multi-agent-forecasting/
- Direct source claims:
  - Multi-agent trajectory forecasting benefits from jointly modeling time and social dimensions.
  - Agent-aware attention preserves agent identities while allowing cross-agent temporal interaction.
- Repo implications/inferences:
  - If adding action chunks/history, use agent-time tokens or compact per-agent temporal summaries.
  - Preserve agent identity only within trajectory history, not as absolute learned IDs that break permutation generalization.

### VectorNet

- Authors/org: Jiyang Gao, Chen Sun, Hang Zhao, Yi Shen, Dragomir Anguelov, Congcong Li, Cordelia Schmid; Waymo/Google, CVPR 2020.
- Link: https://waymo.com/research/vectornet-encoding-hd-maps-and-agent-dynamics-from-vectorized-representation/
- Direct source claims:
  - Vectorized representation of maps and trajectories avoids lossy raster rendering and heavy ConvNet encoding.
  - Hierarchical graph network models local polyline structure and global interactions, saving parameters/FLOPs.
- Repo implications/inferences:
  - Long-term map representation should consider obstacle boundary/segment tokens or SDF contour tokens, not only raster patches.
  - Near-term local patches are fine; obstacle-heavy maps may need vector/map-token cross-attention.

### ALiBi / RoPE / relative position alternatives

- Links:
  - ALiBi: https://huggingface.co/papers/2108.12409
  - RoPE: https://huggingface.co/papers/2104.09864
- Direct source claims:
  - ALiBi biases attention scores by distance for length extrapolation.
  - RoPE encodes positional information with rotations and yields relative position behavior.
- Repo implications/inferences:
  - For 2D metric agent sets, learned edge-feature attention bias is more direct than sequence ALiBi/RoPE.
  - A radial distance bias inspired by ALiBi can regularize local attention and extrapolate to larger maps/agent counts.

## Flow Matching / Rectified Flow

### Flow Matching for Generative Modeling

- Authors/org: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matthew Le; ICLR 2023.
- Links:
  - https://openreview.net/forum?id=PqvMRDCJT9t
  - https://iclr.cc/virtual/2023/poster/11309
- Direct source claims:
  - Flow Matching trains continuous normalizing flows by simulation-free regression of vector fields along fixed probability paths.
  - Compatible with Gaussian paths and OT displacement interpolation.
  - OT paths can improve training/sampling efficiency and generalization.
- Repo implications/inferences:
  - Existing loss in `train_continuous.py` matches the simple rectified/linear interpolation spirit.
  - For control, flow target should be defined over action/velocity chunks or preferred-control distributions, then shielded at execution.

### Flow Straight and Fast / Rectified Flow

- Authors/org: Xingchao Liu, Chengyue Gong, Qiang Liu; ICLR 2023.
- Links:
  - https://iclr.cc/virtual/2023/oral/12626
  - https://huggingface.co/papers/2209.03003
- Direct source claims:
  - Rectified flow learns ODEs to follow straight paths connecting samples from two distributions via simple least squares.
  - Straight flows can be simulated accurately with coarse Euler discretization and even one step in some empirical settings.
- Repo implications/inferences:
  - Repo's low integration-step flow inference is conceptually plausible.
  - Acceptance should still be empirical: compare `steps=1/2/3/5`, because safety-shielded control is not image generation.
