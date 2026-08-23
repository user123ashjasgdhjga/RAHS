import json
from multiprocessing import Pool
import numpy as np
import time

from applications import APPLICATIONS
from gavel import GavelPolicy
from job import Job
from sia import SiaPolicy
from sia_fix import SiaFixPolicy
from utils import JobInfo, NodeInfo
import simulator_config as sim_config


class Cluster(object):
    # num_nodes, ngpus_per_node are dicts with key=cluster_name
    def __init__(self, workload, policy, num_nodes, ngpus_per_node, max_physical_nodes):
        print(f"Simulated cluster config: \nnnodes: {num_nodes}, ngpus_per_node:{ngpus_per_node}")
        # workload is a pandas dataframe
        self.workload = workload
        # policy is a Policy object
        self.policy = policy
        self.num_nodes, self.ngpus_per_node = num_nodes, ngpus_per_node
        self.current_time = 0
        self.clusters = list(self.num_nodes.keys())
        self.max_physical_nodes = max_physical_nodes
        self.rescale_events, self.migrate_events = [], []

        # ===== 新增：双层调度统计信息 =====
        self.macro_scheduling_rounds = 0
        self.micro_adjustments_count = 0
        self.model_correction_history = []

        # create inverse map like {app-name: {cluster_name: cluster_app}}
        self.cluster_applications = dict()
        for cluster_name, cluster_apps in APPLICATIONS.items():
            for app_name, app in cluster_apps.items():
                if app_name not in self.cluster_applications:
                    self.cluster_applications[app_name] = dict()
                self.cluster_applications[app_name][cluster_name] = app

        if isinstance(policy, SiaPolicy):
            # cache_speedups = isinstance(policy, PolluxPolicy)
            cache_speedups = False
            self.jobs = []
            adaptive_jobnames = list()
            rigid_jobnames = list()
            scale_gpu_jobnames = list()
            for row in workload.itertuples():
                category = row.category if 'category' in workload.columns else None
                cur_job = Job(row.name, self.cluster_applications[row.application],
                              row.time, cache_speedups=cache_speedups,
                              target_num_replicas=row.num_replicas,
                              target_batch_size=row.batch_size, category=category)
                if category is not None:
                    if category == 'scale-gpus':
                        scale_gpu_jobnames.append(cur_job.name)
                        cur_job.disable_bsz_tuning()
                    elif category == 'adaptive':
                        adaptive_jobnames.append(cur_job.name)
                    elif category == 'rigid':
                        rigid_jobnames.append(cur_job.name)
                        cur_job.disable_bsz_tuning()
                self.jobs.append(cur_job)
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
            print(f"Initialized job state for {len(self.jobs)} jobs")
            print(f"Scaling only GPU count (scale-gpu): {scale_gpu_jobnames}")
            print(f"Scaling bsz, GPU count (adaptive): {adaptive_jobnames}")
            print(f"Fixed bsz, GPU count (rigid): {rigid_jobnames}")
            # enable bsz adaptation for Sia by default
            self.enable_bsz_tuning = True
        elif isinstance(policy, GavelPolicy):
            # change the target num_replicas to fit into the smallest cluster
            smallest_cluster_size = min(
                [self.num_nodes[k] * self.ngpus_per_node[k] for k in self.num_nodes.keys()])
            self.jobs = [Job(row.name, self.cluster_applications[row.application], row.time,
                             target_num_replicas=row.num_replicas, target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            # disable bsz adaptation for Gavel
            self.enable_bsz_tuning = False
            for job in self.jobs:
                job.fix_minibatch_size()
        elif isinstance(policy, SiaFixPolicy):
            smallest_cluster_size = min(
                [self.num_nodes[k] * self.ngpus_per_node[k] for k in self.num_nodes.keys()])
            self.jobs = [Job(row.name, self.cluster_applications[row.application],
                             row.time, cache_speedups=False,
                             target_num_replicas=min(
                                 row.num_replicas, smallest_cluster_size),
                             target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
        else:
            assert False, f"unsupported policy {policy.__class__.__name__}"
        self.allocations = {}
        self.logs = []
        self.utility = []
        self.last_policy_runtime_ms = 0.0  # <==== 新增：用于记录上一轮策略运行耗时 (毫秒)
        self.compute_throughput_ratios(init=False)
        self.compute_throughputs()

        # multi-processing for job.step() in parallel
        self.process_pool = Pool(sim_config.num_parallel_jobs) if sim_config.num_parallel_jobs > 1 else None

    # disable bsz tuning for MIP
    def disable_bsz_tuning(self):
        self.enable_bsz_tuning = False
        for job in self.jobs:
            job.disable_bsz_tuning()


    def calculate_cross_job_interference(self):



        node_occupancy = {}

        # 获取当前正在运行的作业
        active_jobs = [j for j in self.jobs if j.submission_time <= self.current_time
                       and j.completion_time is None and j.placement]

        for job in active_jobs:
            cluster_name = job.current_cluster
            # job.placement 存储的是该作业占用的节点索引，例如 [0, 0, 1, 1]
            unique_nodes = set(job.placement)
            for node_id in unique_nodes:
                key = (cluster_name, node_id)
                if key not in node_occupancy:
                    node_occupancy[key] = set()
                node_occupancy[key].add(job.name)

        # 2. 计算每个作业的干扰惩罚
        penalty_per_job = getattr(sim_config, 'interference_penalty', 0.15)

        for job in active_jobs:
            cluster_name = job.current_cluster
            unique_nodes = set(job.placement)

            # 寻找该作业所在的最“拥挤”的节点
            max_others = 0
            for node_id in unique_nodes:
                others_count = len(node_occupancy[(cluster_name, node_id)]) - 1
                if others_count > max_others:
                    max_others = others_count

            # 干扰因子公式：1.0 - (额外作业数 * 惩罚系数)
            # 结果越小，吞吐量降级越严重
            factor = 1.0 - (max_others * penalty_per_job)
            job.current_interference_factor = max(0.2, factor) # 设定最低保障性能为 20%

            # 可选：如果是 debug 模式，打印干扰显著的作业
            if factor < 1.0 and getattr(sim_config, 'log_exploration_verbose', False):
                print(f"  [干扰监控] 作业 {job.name} 在 {cluster_name} 节点上与其他 {max_others} 个作业竞争，性能系数: {factor:.2f}")

    # simulate one scheduling round
    def step(self, seconds=60):
        # ===== 新增：双层调度标记 =====
        print(f"\n{'=' * 70}")
        print(f"[集群调度] 时间: {self.current_time}s, 调度周期: {seconds}s")
        print(f"{'=' * 70}")

        # ==========================================================
        # [新增] 压力测试：计算跨作业干扰因子 (Cross-Job Interference)
        # ==========================================================
        if getattr(sim_config, 'enable_cross_job_interference', False):
            self.calculate_cross_job_interference()
        else:
            # 如果未启用，重置所有作业的干扰因子为 1.0 (无损失)
            for job in self.jobs:
                job.current_interference_factor = 1.0
            # ==========================================================


        # advance job state for `seconds` seconds
        start_t = time.time()
        if self.process_pool:
            self.step_jobs_parallel(seconds)
        else:
            self.step_jobs_sequential(seconds)
        end_t = time.time()
        print(f"[性能统计] 作业执行时间: {(end_t - start_t) * 1000:.2f}ms")

        # ===== 新增：统计微观层调整 =====
        self.collect_micro_adjustment_statistics()

        # invoke scheduler policy to optimize allocations for next round
        # returns a list of changes to allocations as rescale and migrate events for logging
        step_rescale_events, step_migrate_events = self.optimize_policy()

        # log cluster state and metrics
        self.rescale_events.append(step_rescale_events)
        self.migrate_events.append(step_migrate_events)
        self.step_log()

        # ===== 新增：双层调度摘要 =====
        self.print_two_level_scheduling_summary()

    def step_jobs_sequential(self, seconds=60):
        # 挂载配置中的开关
        check_interference = getattr(sim_config, 'enable_cross_job_interference', False)
        interference_penalty = getattr(sim_config, 'interference_penalty', 0.15)

        # 统计每个物理节点 (node_id) 上正在运行的不同作业数量
        node_job_counts = {}
        if check_interference:
            for job in self.jobs:
                # 仅统计正在运行的作业
                if job.start_time is not None and job.completion_time is None and len(job.placement) > 0:
                    # 假设 job.placement 是一个 tuple，代表每个节点分配的 GPU 数量。
                    # 我们需要找到它具体映射到哪台机器。为了简化，如果一个作业有分布式的 placement，
                    # 认为它对其占用的所有拓扑产生了占用度。
                    for node_idx, node_gpus in enumerate(job.placement):
                        if node_gpus > 0:
                            # 简单模拟：以 cluster_name + node_idx 作为唯一物理节点标识
                            unique_node_id = f"{job.current_cluster}_{node_idx}"
                            node_job_counts[unique_node_id] = node_job_counts.get(unique_node_id, 0) + 1

        for job in self.jobs:
            interference = 0.0
            if check_interference and len(job.placement) > 0:
                max_colocated_jobs = 1
                for node_idx, node_gpus in enumerate(job.placement):
                    if node_gpus > 0:
                        unique_node_id = f"{job.current_cluster}_{node_idx}"
                        max_colocated_jobs = max(max_colocated_jobs, node_job_counts.get(unique_node_id, 1))

                # 如果同一节点上有大于1个作业，发生干扰
                if max_colocated_jobs > 1:
                    # 最高干扰上限设为 80% 损耗
                    interference = min(0.8, (max_colocated_jobs - 1) * interference_penalty)

            # 注意：此处确保 multi_simulator.py 没有强制调用 step_jobs_parallel
            # 因为并行计算由于内存隔离很难传递实时的 interference 状态
            job.step(seconds, interference=interference)

        self.current_time += seconds

    # WARNING:: disables batch size & interference check (see other version above)
    def step_jobs_parallel(self, seconds=60):
        self.jobs = self.process_pool.starmap(_helper_step_job_single, [(job, seconds) for job in self.jobs])
        self.current_time += seconds

    def optimize_policy(self):
        job_infos = self.get_job_infos()
        policy_optim_time, alloc_event_time = 0, 0
        if not job_infos:
            return [], []


        # # ===== 新增：宏观调度轮次统计 =====
        # if isinstance(self.policy, SiaPolicy):
        #     self.macro_scheduling_rounds += 1
        #     print(f"\n[宏观调度器] 第 {self.macro_scheduling_rounds} 轮全局优化")

        # Optimize allocations.
        node_infos = self.get_node_infos()
        self.allocations = {
            k: v for k, v in self.allocations.items() if k in job_infos}

        # ===== 新增：为SiaPolicy传递实际Job对象引用 =====
        if isinstance(self.policy, SiaPolicy):
            # 更新policy中的实际Job对象引用
            for job in self.jobs:
                if job.submission_time <= self.current_time and job.completion_time is None:
                    self.policy.actual_jobs[job.name] = job

        # run scheduling policy
        p_start_t = time.time()
        allocations = self.policy.optimize(
            job_infos, node_infos, self.allocations)
        policy_optim_time = time.time() - p_start_t

        self.last_policy_runtime_ms = policy_optim_time * 1000  # <==== 新增：保存策略运行时间（毫秒）

        # collect any change in allocations
        # rescale: change in ngpus, migrate: change in ngpus and/or cluster
        p_start_t = time.time()
        rescale_events, migrate_events = [], []
        for job in self.jobs:
            if job.submission_time <= self.current_time and job.completion_time is None:
                job.contention.append(len(job_infos))
                if isinstance(self.policy, SiaPolicy) or isinstance(self.policy, GavelPolicy) or isinstance(self.policy,
                                                                                                            SiaFixPolicy):
                    cluster, alloc = allocations.get(job.name, (None, ()))
                    if isinstance(self.policy, SiaFixPolicy) or isinstance(self.policy, GavelPolicy):
                        assert len(alloc) == 0 or len(
                            alloc) == job.target_num_replicas, f'{job.name} target_num_replicas: {job.target_num_replicas} alloc: {alloc}'
                    # change in resources
                    if allocations.get(job.name) != self.allocations.get(job.name):
                        placement = []
                        for i in range(len(alloc)):
                            if i == 0 or alloc[i] != alloc[i - 1]:
                                placement.append(1)
                            else:
                                placement[-1] += 1
                        # fix placement if bad
                        nnodes = len(placement)
                        nreplicas = sum(placement)
                        if cluster is not None and nnodes > self.max_physical_nodes[cluster]:
                            n_reps_per_node = int(np.ceil(
                                nreplicas / self.max_physical_nodes[cluster]))
                            new_placement = []
                            while nreplicas > 0:
                                n_reps = min(n_reps_per_node, nreplicas)
                                new_placement.append(n_reps)
                                nreplicas -= n_reps
                            print(
                                f"WARNING:: FIXING PLACEMENT FOR LARGE CLUSTER: {cluster}, old placement:{placement}, new/fixed placement: {new_placement}")
                            placement = new_placement
                        if job.current_cluster != cluster:
                            # migrated to another cluster
                            migrate_events.append(job.migrate(cluster, placement))
                        elif self.allocations.get(job.name) != alloc:
                            # reallocated within same cluster
                            rescale_events.append(job.reallocate(placement))
                else:
                    assert False, "other policies not implemented"
        # make a copy of allocations
        self.allocations = allocations
        alloc_event_time = time.time() - p_start_t
        print(f"[性能统计] 策略优化: {policy_optim_time * 1000:.2f}ms, 资源重分配: {alloc_event_time * 1000:.2f}ms")
        if sim_config.log_sched_decisions:
            print(f"Rescale events: {rescale_events}")
            print(f"Migrate events: {migrate_events}")
        return rescale_events, migrate_events

    # ===== 新增：收集微观层调整统计 =====
    def collect_micro_adjustment_statistics(self):
        """
        收集所有作业的微观层批次大小调整统计信息
        """
        total_adjustments = 0
        jobs_with_adjustments = 0

        for job in self.jobs:
            if job.submission_time <= self.current_time and job.completion_time is None:
                bsz_stats = job.get_batch_size_statistics()
                if bsz_stats and bsz_stats['samples'] > 0:
                    total_adjustments += bsz_stats['samples']
                    jobs_with_adjustments += 1

        self.micro_adjustments_count += total_adjustments

        if jobs_with_adjustments > 0:
            print(f"\n[微观层统计] 本周期微调次数: {total_adjustments}, "
                  f"涉及作业: {jobs_with_adjustments}")

    # ===== 新增：打印双层调度摘要 =====
    def print_two_level_scheduling_summary(self):
        """
        打印双层调度体系的运行摘要
        """
        if not isinstance(self.policy, SiaPolicy):
            return

        active_jobs = [job for job in self.jobs
                       if job.submission_time <= self.current_time
                       and job.completion_time is None]

        if not active_jobs:
            return

        print(f"\n{'=' * 70}")
        print(f"[双层调度摘要] 时间: {self.current_time}s")
        print(f"{'=' * 70}")
        print(f"宏观调度轮次: {self.macro_scheduling_rounds}")
        print(f"累计微观调整: {self.micro_adjustments_count}")
        print(f"活跃作业数量: {len(active_jobs)}")

        # 统计建议批次大小与实际执行的偏差
        total_deviation = 0
        jobs_with_suggestions = 0

        for job in active_jobs:
            if job.suggested_batch_size is not None:
                bsz_stats = job.get_batch_size_statistics()
                if bsz_stats and bsz_stats['samples'] > 0:
                    deviation_percent = (bsz_stats['deviation'] / job.suggested_batch_size * 100
                                         if job.suggested_batch_size > 0 else 0)
                    total_deviation += deviation_percent
                    jobs_with_suggestions += 1

                    print(f"  {job.name[:20]:20s} | "
                          f"建议: {job.suggested_batch_size:6d} | "
                          f"实际: {bsz_stats['mean']:6.1f} ± {bsz_stats['std']:5.1f} | "
                          f"偏差: {deviation_percent:5.2f}%")

        if jobs_with_suggestions > 0:
            avg_deviation = total_deviation / jobs_with_suggestions
            print(f"\n平均偏差率: {avg_deviation:.2f}%")

        print(f"{'=' * 70}\n")

    def compute_throughput_ratios(self, init=False):
        # throughput conversion ratios
        self.cluster_throughput_ratios = dict()
        for app_name in self.cluster_applications.keys():
            if app_name == "gpt_pmp":
                continue
            app_xput_ratios = dict()
            for cluster in self.clusters:
                ratios = dict()
                for dest_cluster in self.clusters:
                    # ignore any clusters that dont have speedup_fn ready
                    if dest_cluster == cluster:
                        continue
                    elif not init:
                        dest_df = self.cluster_applications[app_name][dest_cluster].placements
                        src_df = self.cluster_applications[app_name][cluster].placements
                        src_num_stages = self.cluster_applications[app_name][cluster].num_stages
                        dest_num_stages = self.cluster_applications[app_name][dest_cluster].num_stages
                        # get slice of placements with min number of stages (=1 for standard DP jobs)
                        if src_num_stages == dest_num_stages == 1:
                            src_df = src_df[src_df.num_replicas == 1]
                            dest_df = dest_df[dest_df.num_replicas == 1]
                            mean_src_atomic_xput = (
                                    src_df.local_bsz.max() / src_df.step_time.max())
                            mean_dest_atomic_xput = (
                                    dest_df.local_bsz.max() / dest_df.step_time.max())
                        else:
                            # PMP jobs
                            src_df = src_df[src_df.num_replicas ==
                                            src_num_stages]
                            dest_df = dest_df[dest_df.num_replicas ==
                                              dest_num_stages]
                            mean_src_atomic_xput = (
                                    src_df.local_bsz.max() / src_df.step_time.max())
                            mean_dest_atomic_xput = (
                                    dest_df.local_bsz.max() / dest_df.step_time.max())
                            print(
                                f"src_cluster: {cluster}, dest_cluster: {dest_cluster}")
                            print(
                                f"mean_src_atomic_xput: {mean_src_atomic_xput}, mean_dest_atomic_xput: {mean_dest_atomic_xput}")
                        # record ratio as (mean xput in src_cluster) / (mean xput in dest_cluster)
                        ratio = mean_src_atomic_xput / mean_dest_atomic_xput
                        ratios[dest_cluster] = ratio
                    else:
                        ratios[dest_cluster] = 1.0
                app_xput_ratios[cluster] = ratios
            self.cluster_throughput_ratios[app_name] = app_xput_ratios
        if sim_config.log_cluster_verbose:
            print(f"Throughput ratios: {self.cluster_throughput_ratios}")

    def compute_throughputs(self):
        # throughput conversion ratios
        self.cluster_throughputs = dict()
        for app_name in self.cluster_applications.keys():
            if app_name == "gpt_pmp":
                continue
            app_xputs = dict()
            for cluster in self.clusters:
                cluster_profiles = self.cluster_applications[app_name][cluster].placements
                cluster_num_stages = self.cluster_applications[app_name][cluster].num_stages
                # standard DP jobs
                if cluster_num_stages == 1:
                    min_profile = cluster_profiles[cluster_profiles.num_replicas == 1]
                    mean_atomic_xput = (min_profile.local_bsz.max() / min_profile.step_time.max())
                # PMP jobs
                else:
                    min_profile = cluster_profiles[cluster_profiles.num_replicas == cluster_num_stages]
                    mean_atomic_xput = (min_profile.local_bsz.max() / min_profile.step_time.max())
                # atomic xput = xput with minimum number of replicas
                app_xputs[cluster] = mean_atomic_xput
            self.cluster_throughputs[app_name] = app_xputs
        if sim_config.log_cluster_verbose:
            print(f"min-GPU throughputs: {self.cluster_throughputs}")

    def step_log(self):
        def get_bsz(job):
            ngpus = int(sum(job.placement))
            num_replicas = ngpus
            # if job is a pipeline, divide by number of stages
            if job.num_stages > 1:
                num_replicas //= job.num_stages
            return job.atomic_bsz * (job.accum_steps + 1) * num_replicas

        # Extract correction factors if using SiaPolicy
        correction_factors = {}
        if hasattr(self.policy, 'throughput_model_correction_factors'):
            correction_factors = self.policy.throughput_model_correction_factors.copy()

        step_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            # ===== 新增：双层调度元数据 =====
            "macro_scheduling_round": self.macro_scheduling_rounds,
            "total_micro_adjustments": self.micro_adjustments_count,
            "policy_runtime_ms": getattr(self, 'last_policy_runtime_ms', 0.0),
            "correction_factors": correction_factors, # <==== Added: phi values
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "cluster": job.current_cluster,
                    "allocation": self.allocations.get(job.name, tuple()),
                    "placement": job.placement,
                    "batch_size": get_bsz(job),
                    "accum_steps": job.accum_steps,
                    "stages": job.num_stages,
                    "submission_time": job.submission_time,
                    "start_time": job.start_time,
                    "completion_time": job.completion_time,
                    "n_avg": np.mean(np.asarray(job.contention)) if job.contention else 0,
                    "category": job.category,
                    # ===== 新增：双层调度作业级统计 =====
                    "suggested_batch_size": job.suggested_batch_size,
                    "batch_size_statistics": job.get_batch_size_statistics(),
                    "recent_throughput_samples": (list(job.recent_throughput_samples[-5:])
                                                  if hasattr(job, 'recent_throughput_samples')
                                                     and job.recent_throughput_samples else []),
                    "micro_adjustment_count": (len(job.actual_batch_size_history)
                                               if hasattr(job, 'actual_batch_size_history') else 0),
                    # ===== 新增：最优批次大小信息 =====
                    "optimal_bsz_info": (job.get_optimal_bsz_for_goodput()
                                        if hasattr(job, 'get_optimal_bsz_for_goodput') else None),
                    "best_bsz_in_current_period": (job.best_bsz_in_period
                                                   if hasattr(job, 'best_bsz_in_period') else None),
                    "best_throughput_in_current_period": (job.best_throughput_in_period
                                                          if hasattr(job, 'best_throughput_in_period') else 0),

                    # ===== 新增：风险调整相关字段 =====
                    "current_sensitivity": (job.current_sensitivity
                                            if hasattr(job, 'current_sensitivity') else 0.0),
                    "sensitivity_history": (job.batch_size_sensitivity_history[-5:]
                                            if hasattr(job, 'batch_size_sensitivity_history')
                                               and job.batch_size_sensitivity_history
                                            else []),
                    "risk_adjusted_goodput": (job.risk_adjusted_goodput
                                              if hasattr(job, 'risk_adjusted_goodput') else 0.0),
                    # <==== Added: original unadjusted goodput for prediction error comparison
                    "predicted_goodput": (job.batch_size_sensitivity_history[-1]['goodput_original']
                                          if hasattr(job, 'batch_size_sensitivity_history') and job.batch_size_sensitivity_history else 0.0)
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        }
        self.logs.append(step_log)

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, SiaPolicy):
                    job_infos[job.name] = self.get_sia_multi_job_info(
                        job)
                elif isinstance(self.policy, GavelPolicy):
                    job_infos[job.name] = self.get_gavel_job_info(job)
                elif isinstance(self.policy, SiaFixPolicy):
                    job_infos[job.name] = self.get_sia_fix_multi_job_info(
                        job)
                else:
                    job_infos[job.name] = self.get_pollux_job_info(job)
        return job_infos

    def get_pollux_job_info(self, job):
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,
                             job.application.max_batch_size // job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_sia_multi_job_info(self, job):
        speedup_fns = {cname: job.get_speedup_fn(
            cname) for cname in self.clusters}
        scale_factor = 2
        max_replicas = {cname: min(max(scale_factor * job.max_profiled_replicas(cname), 1), 64,
                                   job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas={cname: 0 for cname in self.clusters},
            max_replicas=max_replicas,
            preemptible=True,
        )

        # attach used+wasted gpu seconds to job_info
        job_info.used_gpu_seconds = job.used_gpu_seconds
        job_info.wasted_gpu_seconds = job.wasted_gpu_seconds

        # record job category
        job_info.category = job.category
        if job_info.category == "rigid" or job_info.category == "scale-gpus":
            job_info.max_replicas = {
                cname: job.target_num_replicas for cname in self.clusters}

        # treat NCF jobs as rigid with max_replicas = 1
        if 'ncf' in job.applications:
            job_info.max_replicas = {cname: 1 for cname in self.clusters}
            job_info.category = "rigid"

        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time

        app_name = job.name.split('-')[0]

        # throughput conversion ratios
        job_info.cluster_throughput_ratios = self.cluster_throughput_ratios[app_name]
        job_info.cluster_throughputs = self.cluster_throughputs[app_name]

        # Rigid jobs ==> throughput ratios == throughputs of the job
        if job_info.category == "rigid":
            job_info.cluster_throughput_ratios = job.get_throughputs(
                self.ngpus_per_node)
            print(
                f"Rigid job: {job.name}, throughput ratios: {job_info.cluster_throughput_ratios}")

        # if job is currently in rescale stage
        if sim_config.preserve_ckpt_allocs:
            job_info.scaling_underway = job.rescale_time > 0
        else:
            job_info.scaling_underway = False

        # scale units for this job
        job_info.scale_unit = job.get_scale_units()

        return job_info

    def get_sia_fix_multi_job_info(self, job):
        speedup_fns = {cname: job.get_speedup_fn(
            cname) for cname in self.clusters}
        target_replicas = {
            cname: job.target_num_replicas for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=target_replicas,
            max_replicas=target_replicas,
            preemptible=True,
        )

        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time

        app_name = job.name.split('-')[0]

        # throughput conversion ratios
        job_info.cluster_throughput_ratios = self.cluster_throughput_ratios[app_name]

        return job_info

    def get_gavel_job_info(self, job):
        speedup_fns = {cname: job.get_speedup_fn(
            cname) for cname in self.clusters}
        max_replicas = {cname: min(max(2 * job.max_profiled_replicas(cname), 1), 64,
                                   job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas={cname: 0 for cname in self.clusters},
            max_replicas=max_replicas,
            preemptible=True,
        )
        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.applications = job.applications
        job_info.epoch = job.epoch
        job_info.target_batch_size = job.target_batch_size
        job_info.scale_factor = job.target_num_replicas
        job_info.age = self.current_time - job.submission_time
        job_app = list(job.applications.values())[0]
        job_info.cur_step = job_app.get_cur_iteration(
            job.target_batch_size, job.epoch, job.progress)
        job_info.total_steps = job_app.get_iteration(
            job.target_batch_size, job.completion_epoch)
        job_info.remain_steps = max(
            1, job_info.total_steps - job_info.cur_step)
        job_info.rescale_time = job_app.rescale_time
        job_info.submission_time = job.submission_time
        job_info.slowdown_factor = job.calibration_factor
        assert job_app.rescale_time > 0
        return job_info

    def get_node_infos(self, num_nodes=None):
        cluster_node_info = dict()
        for cluster_name in self.clusters:
            cluster_info = {idx: NodeInfo({"nvidia.com/gpu": self.ngpus_per_node[cluster_name]}, preemptible=False)
                            for idx in range(self.num_nodes[cluster_name])}
            cluster_node_info[cluster_name] = cluster_info
        return cluster_node_info

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_jcts(self):
        if len(self.logs) > 0:
            return {
                val["name"]: val["completion_time"] - val["submission_time"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None
            }
        else:
            return {}


def _helper_step_job_single(job, seconds):
    job.step(seconds=seconds, interference=0)
    # important to return job object as all changes are process-local
    return job