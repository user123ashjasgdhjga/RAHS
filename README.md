- # RAHS: A Risk-Aware Hierarchical Scheduler for Heterogeneous GPU Clusters

  RAHS is a novel scheduling system for efficient scheduling of deep learning (DL) jobs in heterogeneous GPU clusters. Through a decoupled macro-micro two-layer scheduling framework, a risk-adjusted utility function, and a feedback-driven batch size refinement mechanism, RAHS effectively addresses the mismatch between scheduling decisions and actual execution states in large-scale heterogeneous clusters.

  ## Project Structure

  ```
  ├── rahs.py              # Core scheduling algorithm implementation
  ├── multi_simulator.py   # Simulator entry point
  ├── cluster.py           # Cluster simulation and resource management
  ├── job.py               # Job modeling and lifecycle management
  ├── goodput.py           # Goodput function modeling
  ├── speedup.py           # Speedup function
  ├── applications.py      # Application configuration and performance trace loading
  └── utils.py             # Utility classes
  ```

  ## Environment Dependencies

  - Python 3.7+
  - cvxpy >= 1.0 
  - numpy
  - scipy
  - pandas
  - autograd
  - rich

  Install dependencies:

  ```bash
  pip install cvxpy numpy scipy pandas autograd rich
  ```

  ## Quick Start

  ### Basic Usage

  ```bash
  python multi_simulator.py <workload.csv> --policy rahs --interval 60
  ```

  ### Common Parameters

  | Parameter         | Description                       | Default |
  | ----------------- | --------------------------------- | ------- |
  | `workload`        | Workload CSV file path (required) | -       |
  | `--policy`        | Scheduling policy                 | rahs    |
  | `--interval`      | Scheduling interval (seconds)     | `60`    |
  | `--cluster_scale` | Cluster scale multiplier          | `None`  |
  | `--output`        | Output log path                   | `None`  |
