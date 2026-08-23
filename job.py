import math
import time
import numpy as np
import random

from applications import APPLICATIONS
import simulator_config as sim_config
from simulator_config import SLOWDOWNS
from goodput import GoodputFunction, GoodputFunctionPMP, fit_perf_params, PerfParams, GradParams
from speedup import SpeedupFunction, UncachedSpeedupFunction
from utils import JobInfo, NodeInfo
from collections import defaultdict

MAX_SLOWDOWN = 15


class Job(object):
    def __init__(self, name, applications, submission_time,
                 target_num_replicas=None, target_batch_size=None,
                 cache_speedups=False, h_unaware=False, category=None):
        ## Job attributes
        self.name = name
        self.app_name = list(applications.values())[0].name
        # applications is a dict w/ key=cluster_name, val=cluster_specific_App
        self.applications = applications
        self.category = category

        if self.category == "rigid":
            self.cluster_throughputs = None

        # submission_time: when it was submitted
        self.submission_time = submission_time
        # target_num_replicas: requested num replicas
        self.target_num_replicas = target_num_replicas
        # target_batch_size: requested batch size
        self.target_batch_size = target_batch_size
        # switch to control bsz tuning
        self.enable_bsz_tuning = True

        # switch to control if job is heterogeneity-aware
        self.h_unaware = h_unaware
        # profile is also a dict w/ key=cluster_name, val=cluster_specific_profile
        # if h_unaware is True, profiles is just all profiles merged into one dict
        self.profiles = dict()
        # perf_params is also a dict w/ key=cluster_name, val=cluster_specific_perf_params
        if self.h_unaware:
            self.perf_params = None
        else:
            self.perf_params = dict()
            for cluster in self.applications.keys():
                self.profiles[cluster] = dict()
                self.perf_params[cluster] = None
        # Optimization state
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.reference_app = self.applications[sim_config.calibrate_cluster]

        ## Job state
        # start_time: when it started running
        self.start_time = None
        # completion_time: when it finished running
        self.completion_time = None
        # current_time: current wall-clock time
        self.current_time = 0
        # (overhead) time to checkpoint + restore job
        # >0 if an operation is pending/executing, 0 otherwise
        self.rescale_time = 0
        self.max_rescale_time = self.reference_app.rescale_time
        # N-tuple: (x_1, x_2, ..., x_N) where x_i is the number of replicas in node i
        self.placement = ()
        # GPU-seconds allocated
        self.attained_service = 0
        # GPU-seconds used by job
        self.used_gpu_seconds = 0
        # GPU-seconds wasted by job
        self.wasted_gpu_seconds = 0

        # number of jobs contending for resources (one value per round)
        self.contention = []
        # number of times the job has been restarted
        self.num_restarts = None
        # number of times the job has been migrated between different clusters
        # note: num_migrations <= num_restarts
        self.num_migrations = None
        # events corresponding to job state changes
        self.bsz_update_events = []
        self.rescale_events = []
        self.migrate_events = []
        # num_stages > 1 for Pipeline-Model-Parallelism(PMP) jobs
        self.num_stages = self.reference_app.num_stages

        # cluster that this job is allocated to
        self.current_cluster = None
        self.speedup_fn_class = SpeedupFunction if cache_speedups else UncachedSpeedupFunction

        # shockwave addons
        self.execution_time = 0
        self.epoch_duration = []
        self.completion_epoch = self.reference_app.get_completion_epoch(
            self.target_batch_size) if self.target_batch_size is not None else None

        # calibration factor for this job (calibration step to account for real-world runs being slower than simulator)
        self.calibration_factor = 1 / SLOWDOWNS[self.app_name]
        if sim_config.log_cluster_verbose:
            print(
                f"{self.name}: SpeedupFnClass={self.speedup_fn_class.__name__}, calibration-factor={self.calibration_factor:.2f}")

        # ===== 新增：异构反馈修正因子 (Per-GPU-Type) =====
        # 记录作业当前实际运行的 GPU 类型，用于匹配和更新
        self.current_gpu_type = None
        # 细粒度修正因子字典：key=gpu_type (如 "A100", "T4"), value=correction_factor
        # 使用 defaultdict，使得任意未曾跑过的硬件类型（冷启动），其修正因子自动默认初始化为 1.0，保证调度安全性
        self.correction_factors = defaultdict(lambda: 1.0)

        # ===== 新增：微观层自适应执行器属性 =====
        # 宏观层建议的批次大小（来自中心调度器）
        self.suggested_batch_size = None
        # 当前调度周期内实际执行的批次大小历史记录
        self.actual_batch_size_history = []
        # 当前调度周期开始时间
        self.current_scheduling_period_start = 0
        # 微调周期（秒）- 比宏观调度更频繁，默认10秒进行一次微调
        self.micro_adjustment_interval = 5
        # 上次微调时间
        self.last_micro_adjustment_time = 0
        # 批次大小动态调整范围（相对于建议值的百分比）±15%
        self.batch_size_adjustment_range = 0.25
        # 性能监控数据（用于微调决策）
        self.recent_throughput_samples = []
        self.recent_gradient_variance_samples = []

        # ======= 修改：读取全局配置决定是否开启微观层 =======
        self.enable_micro_adjustment = getattr(sim_config, 'enable_micro_adjustment', True)

        # 是否启用微观层自适应调整
        #self.enable_micro_adjustment = True

        self.consecutive_improvements = 0  # 连续改进次数
        self.consecutive_degradations = 0  # 连续性能下降次数
        # 新增：自适应调整步长
        self.adjustment_step_size = 1.05  # 初始步长5%
        self.min_step_size = 1.02  # 最小步长2%
        self.max_step_size = 1.15  # 最大步长15%
        self.best_batch_size_performance = {}  # 记录不同批次大小的性能

        # ===== 新增：最优批次大小追踪 =====
        # 当前调度周期内的最优批次大小及其性能
        self.best_bsz_in_period = None
        self.best_throughput_in_period = 0.0
        self.best_bsz_timestamp = 0
        # 历史最优批次大小记录（用于传递给goodput模块）
        self.optimal_bsz_history = []  # [(timestamp, bsz, throughput, num_replicas)]

        # ===== 新增：风险调整相关属性 =====
        # 批次大小敏感度历史
        self.batch_size_sensitivity_history = []
        # 当前配置的敏感度
        self.current_sensitivity = 0.0
        # 风险调整后的goodput
        self.risk_adjusted_goodput = 0.0

    # optimizes perf params given an input perf profile
    # perf_profile is a dict w/ key=(num_nodes, num_replicas, local_bsz), val=(step_time, sync_time)
    # returns a PerfParams object
    def optimize_perf_params(self, perf_profile):
        if perf_profile is None:
            return None
        num_nodes = np.array([key[0] for key in perf_profile])
        num_replicas = np.array([key[1] for key in perf_profile])
        local_bsz = np.array([key[2] for key in perf_profile])
        step_time = np.array([val[0] for val in perf_profile.values()])
        sync_time = np.array([val[1] for val in perf_profile.values()])
        compute_time = step_time - sync_time

        perf_params = fit_perf_params(
            num_nodes, num_replicas, local_bsz, compute_time, step_time)
        return perf_params

    def seed_profiles(self, max_num_nodes, max_num_replicas):
        print(f"Seeding profiles for job: {self.name}")
        for cluster, cluster_app in self.applications.items():
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]

            # add placements data
            if max_num_nodes > 0:
                placements_selector = (cluster_app.placements.num_nodes <= max_num_nodes) & (
                        cluster_app.placements.num_replicas <= max_num_replicas)
                df = cluster_app.placements[placements_selector]
            else:
                df = cluster_app.placements

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
            ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                self.profiles[cluster][num_nodes[i], num_replicas[i],
                local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            if max_num_nodes > 0:
                scalability_selector = (cluster_app.scalability.num_nodes <= max_num_nodes) & (
                        cluster_app.scalability.num_replicas <= max_num_replicas)
                df = cluster_app.scalability[scalability_selector]
            else:
                df = cluster_app.scalability

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
            ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                if profile_key not in profile:
                    profile[profile_key] = step_time[i], sync_time[i]

            # update perf params for cluster
            self.perf_params[cluster] = self.optimize_perf_params(profile)

    def seed_profiles_rigid(self, cluster_ngpus_per_node):
        print(
            f"Seeding profiles for job: {self.name}, target bsz: {self.target_batch_size}, target num replicas: {self.target_num_replicas}")
        for cluster in cluster_ngpus_per_node.keys():
            cluster_app = self.applications[cluster]
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]
            job_num_nodes = np.ceil(self.target_num_replicas /
                                    cluster_ngpus_per_node[cluster])
            job_num_replicas = self.target_num_replicas

            # add placements data if exists
            placements_selector = (cluster_app.placements.num_nodes == job_num_nodes) & (
                    cluster_app.placements.num_replicas <= job_num_replicas)
            df = cluster_app.placements[placements_selector]
            if len(df) == 0:
                print(f"No placements data for job: {self.name}, \
                        cluster: {cluster}, num_nodes: {job_num_nodes}, num_replicas: {job_num_replicas}")
            else:
                print(
                    f"Num placement profiles: {len(df)} for job: {self.name}")
                num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
                ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
                for i in range(len(num_nodes)):
                    self.profiles[cluster][num_nodes[i], num_replicas[i],
                    local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            scalability_selector = (cluster_app.scalability.num_nodes == job_num_nodes) & (
                    cluster_app.scalability.num_replicas <= job_num_replicas)
            df = cluster_app.scalability[scalability_selector]
            if len(df) == 0:
                print(f"No scalability data for job: {self.name}, \
                        cluster: {cluster}, num_nodes: {job_num_nodes}, num_replicas: {job_num_replicas}")
            else:
                print(
                    f"Num scalability profiles: {len(df)} for job: {self.name}")
                num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
                ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
                for i in range(len(num_nodes)):
                    profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                    if profile_key not in profile:
                        profile[profile_key] = step_time[i], sync_time[i]
            if not profile:
                print(
                    f"WARNING: No data for job: {self.name}, cluster: {cluster}")
                continue

            # compute perf params for cluster
            self.perf_params[cluster] = self.optimize_perf_params(profile)
            print(
                f"Initialized goodput fn for job: {self.name}, cluster: {cluster}")

    # returns the maximum number of replicas profiled for a given cluster
    def max_profiled_replicas(self, cluster_name=None):
        max_val = 0
        if not cluster_name and self.h_unaware:
            max_val = max((k[1] for k in self.profiles), default=0)
        if cluster_name in self.profiles:
            max_val = max((k[1]
                           for k in self.profiles[cluster_name]), default=0)
        return max_val

    def get_goodput_fn(self, cluster_name=None):
        app = self.applications[cluster_name if cluster_name else "aws"]
        if self.h_unaware:
            perf_params, grad_params = self.perf_params, self.grad_params
        else:
            perf_params, grad_params = self.perf_params[cluster_name], self.grad_params

        # no throughput model yet
        if grad_params is None or perf_params is None:
            return None
        elif app.num_stages > 1:
            return GoodputFunctionPMP(perf_params, grad_params, app.init_batch_size, app.num_stages,
                                      app.num_microbatches)
        else:
            return GoodputFunction(perf_params, grad_params, app.init_batch_size)

    def get_speedup_fn(self, cluster_name=None):
        if self.h_unaware:
            if self.grad_params is None:
                return lambda n, r: r
        else:
            perf_params = self.perf_params[cluster_name]
            if self.grad_params is None or perf_params is None:
                return None
        app = self.applications[cluster_name if cluster_name else "aws"]
        max_batch_size = app.max_batch_size if self.target_batch_size is None else self.target_batch_size
        bsz_range = (app.min_local_bsz, app.max_local_bsz)
        return self.speedup_fn_class(self.get_goodput_fn(cluster_name), max_batch_size,
                                     bsz_range, accumulation=True, tune_bsz=self.enable_bsz_tuning)

    # returns throughput of job for different GPU types in examples/sec
    def get_throughputs(self, cluster_ngpus_per_node):
        if self.category != "rigid":
            print(
                f"ERROR:: invalid job category: {self.category} for get_throughput call")
        # cluster throughputs are not computed
        if not self.cluster_throughputs:
            self.cluster_throughputs = dict()
            for cname, cngpus_per_node in cluster_ngpus_per_node.items():
                if self.target_num_replicas <= cngpus_per_node:
                    placement = [self.target_num_replicas]
                else:
                    num_whole_nodes = self.target_num_replicas // cngpus_per_node
                    placement = [cngpus_per_node] * num_whole_nodes
                    need_partial = self.target_num_replicas % cngpus_per_node != 0
                    if need_partial:
                        placement.append(
                            self.target_num_replicas % cngpus_per_node)
                self.cluster_throughputs[cname] = self.applications[cname].get_throughput_with_accum(
                    placement, self.target_batch_size) * self.calibration_factor
            print(
                f"Initialized cluster throughputs for job: {self.name} = {self.cluster_throughputs}")
        else:
            print(
                f"Using cached cluster throughputs for job: {self.name} = {self.cluster_throughputs}")
        return {cname: self.cluster_throughputs[cname] for cname in cluster_ngpus_per_node}

    def get_scale_units(self):
        self.scale_units = {
            cluster: app.num_stages for cluster, app in self.applications.items()}
        if sim_config.sia_log_goodputs:
            print(f"Scale units for job: {self.name} = {self.scale_units}")
        return self.scale_units

    # fixes batch size to obey memory/profiling constraints
    # needed for gavel
    def fix_minibatch_size(self):
        print(f"WARNING:: bypassing fix for minibatch size")
        return
        if self.target_num_replicas is not None and self.target_batch_size is not None:
            max_atomic_bsz = math.ceil(
                self.target_batch_size / self.target_num_replicas - 1e-8)
            for cluster, cluster_app in self.applications.items():
                if self.target_num_replicas in cluster_app.placements.num_replicas.values:
                    df = cluster_app.placements[cluster_app.placements.num_replicas ==
                                                self.target_num_replicas]
                    new_bsz = int(min(max_atomic_bsz, df.local_bsz.max()))
                    if new_bsz < max_atomic_bsz:
                        print(
                            f"{self.name}: correcting atomic_bsz: {max_atomic_bsz} -> {new_bsz}")
                        max_atomic_bsz = new_bsz
            target_batch_size = self.target_num_replicas * max_atomic_bsz
            self.target_batch_size = min(
                self.target_batch_size, target_batch_size)

    # call this function to disable bsz tuning
    def disable_bsz_tuning(self):
        self.enable_bsz_tuning = False
        print(f"Disabled bsz tuning for job: {self.name}")

    # given a placement, computes per-GPU batch size, # accumulations
    # uses user-supplied batch size if bsz tuning is disabled
    # NOTE: if bsz tuning disabled ==> the job scales using strong scaling
    def update_local_bsz(self, placement):
        if self.current_cluster is None:
            assert False, "updating local bsz before assigning cluster"
        app = self.applications[self.current_cluster]
        if app.num_stages > 1:
            self.atomic_bsz = 1
            self.accum_steps = app.num_microbatches
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        # use target batch size by default
        batch_size = self.target_batch_size

        # if bsz tuning is enabled, reset batch size and optimize again
        if self.enable_bsz_tuning:
            # print(f"update_local_bsz: bsz tuning ****enabled**** for job: {self.name}")
            batch_size = None
        else:
            # print(f"update_local_bsz: bsz tuning disabled for job: {self.name}")
            pass

        perf_params = self.perf_params if self.h_unaware else self.perf_params[
            self.current_cluster]
        grad_params = self.grad_params
        max_local_bsz = app.get_max_local_bsz(placement)

        # handle PMP jobs gracefully
        if 'pmp' in app.name:
            # obtain pipeline params from app directly
            self.atomic_bsz, self.accum_steps = app.microbatch_size, (
                    app.num_microbatches - 1)
            num_stages = app.num_stages
            # each replica runs on num_stages GPUs
            num_replicas = num_replicas // num_stages
            # obtain global bsz using num_replicas
            bsz_per_replica = self.atomic_bsz * (self.accum_steps + 1)
            batch_size = bsz_per_replica * num_replicas
            max_local_bsz = self.atomic_bsz

        # initial batch size if job has not yet started running
        if batch_size is None and (grad_params is None or perf_params is None):
            batch_size = max(app.init_batch_size,
                             app.min_local_bsz * num_replicas)
        # if we have a goodput function, use it to optimize batch size
        if batch_size is None:
            goodput_fn = self.get_goodput_fn(self.current_cluster)
            # use standard goodput function if exists
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, max_local_bsz), accumulation=True)
        else:
            # otherwise, use the batch size specified by the user
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(
                local_bsz / (self.accum_steps + 1) - 1e-8)

        # correct self.atomic_bsz to take into account memory constraints
        if num_replicas in app.placements.num_replicas.values and 'gpt' not in app.name:
            df = app.placements[app.placements.num_replicas == num_replicas]
            new_bsz = int(min(self.atomic_bsz, df.local_bsz.max()))
            if new_bsz < self.atomic_bsz:
                print(
                    f"WARNING #------>{self.name}: correcting atomic_bsz: {self.atomic_bsz} -> {new_bsz}")
                self.atomic_bsz = new_bsz

        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))
        # print(f"update_local_bsz({self.name}): atomic_bsz={self.atomic_bsz}, accum_steps={self.accum_steps},num_replicas={num_replicas}, batch_size={batch_size}")

    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        assert self.current_cluster is not None, "current_cluster is None??"



        self.grad_params = (grad_sqr, grad_var)
        if self.h_unaware:
            profile = self.profiles
        else:
            profile = self.profiles[self.current_cluster]

        if (num_nodes, num_replicas, local_bsz) in profile:
            return

        if self.h_unaware:
            self.profiles[num_nodes, num_replicas,
            local_bsz] = step_time, sync_time
        else:
            self.profiles[self.current_cluster][num_nodes,
            num_replicas, local_bsz] = step_time, sync_time

        # get PerfParams for these profiles
        perf_params = self.optimize_perf_params(profile)
        if self.h_unaware:
            self.perf_params = perf_params
        else:
            self.perf_params[self.current_cluster] = perf_params

    # ===== 新增：微观层自适应批次大小微调方法 =====
    def micro_adjust_batch_size(self, current_time):
        """
        微观层高频微调批次大小
        围绕宏观层建议值进行自适应调整

        核心思想：
        1. 在宏观层建议值的±15%范围内进行微调
        2. 基于最近的吞吐量趋势进行决策
        3. 吞吐量下降时减小批次大小，吞吐量稳定或上升时增大批次大小
        """
        # 如果未启用微调或没有建议值，直接返回
        if not self.enable_micro_adjustment or self.suggested_batch_size is None:
            return

        # 如果未到微调间隔时间，跳过
        if current_time - self.last_micro_adjustment_time < self.micro_adjustment_interval:
            return

        # 计算允许的批次大小范围（围绕建议值的±15%）
        min_bsz = int(self.suggested_batch_size * (1 - self.batch_size_adjustment_range))
        max_bsz = int(self.suggested_batch_size * (1 + self.batch_size_adjustment_range))

        # 确保至少有最小批次大小
        if self.current_cluster is not None:
            app = self.applications[self.current_cluster]
            min_bsz = max(min_bsz, app.min_local_bsz)
            max_bsz = min(max_bsz, app.max_batch_size)

        # 基于最近的性能指标进行微调决策
        if len(self.recent_throughput_samples) >= 3:
            recent_avg_throughput = np.mean(self.recent_throughput_samples[-3:])

            # 如果有足够的历史数据，比较趋势
            if len(self.recent_throughput_samples) >= 5:
                prev_avg_throughput = np.mean(self.recent_throughput_samples[-5:-2])

                # 吞吐量下降超过5%，减小批次大小以提高稳定性
                if recent_avg_throughput < prev_avg_throughput * 0.99:
                    new_target = max(min_bsz, int(self.target_batch_size * 0.95))
                    if sim_config.log_cluster_verbose:
                        print(f"[微观层微调] {self.name}: 吞吐量下降 "
                              f"({prev_avg_throughput:.2f} -> {recent_avg_throughput:.2f}), "
                              f"减小批次大小 {self.target_batch_size} -> {new_target}")
                # 吞吐量稳定或上升，尝试增大批次大小以提高效率
                else:
                    new_target = min(max_bsz, int(self.target_batch_size * 1.05))
                    if sim_config.log_cluster_verbose:
                        print(f"[微观层微调] {self.name}: 吞吐量稳定/上升 "
                              f"({prev_avg_throughput:.2f} -> {recent_avg_throughput:.2f}), "
                              f"增大批次大小 {self.target_batch_size} -> {new_target}")

                self.target_batch_size = new_target

        # # 基于最近的性能指标进行微调决策
        # if len(self.recent_throughput_samples) >= 3:
        #     recent_avg_throughput = np.mean(self.recent_throughput_samples[-3:])
        #     recent_avg_grad_var = np.mean(self.recent_gradient_variance_samples[-3:]) if len(
        #         self.recent_gradient_variance_samples) >= 3 else 0
        #
        #
        #     # 策略2：基于性能趋势的利用策略
        #     if len(self.recent_throughput_samples) >= 5:
        #         prev_avg_throughput = np.mean(self.recent_throughput_samples[-5:-2])
        #         throughput_change_rate = (recent_avg_throughput - prev_avg_throughput) / (prev_avg_throughput + 1e-6)
        #
        #         # 计算梯度质量指标（方差越小，质量越好）
        #         grad_quality_score = 1.0 / (recent_avg_grad_var + 1.0)
        #
        #         # 决策逻辑
        #         if throughput_change_rate < -0.05:  # 吞吐量下降超过5%
        #             self.consecutive_degradations += 1
        #             self.consecutive_improvements = 0
        #
        #             # 快速响应：连续下降时加大调整力度
        #             if self.consecutive_degradations >= 2:
        #                 self.adjustment_step_size = min(self.adjustment_step_size * 1.1, self.max_step_size)
        #                 print(
        #                     f"[微观层快速响应] {self.name}: 性能持续下降，加大调整步长到 {self.adjustment_step_size:.3f}")
        #
        #             # 吞吐量下降：尝试减小批次大小（提高迭代频率）
        #             # 但如果梯度质量很好，可能是其他原因导致的下降
        #             if grad_quality_score > 0.5:  # 梯度质量好
        #                 new_target = int(self.target_batch_size / self.adjustment_step_size)
        #             else:  # 梯度质量差，可能需要增大批次
        #                 new_target = int(self.target_batch_size * self.adjustment_step_size)
        #
        #             if sim_config.log_cluster_verbose:
        #                 print(f"[微观层微调] {self.name}: 吞吐量下降 "
        #                       f"({prev_avg_throughput:.2f} -> {recent_avg_throughput:.2f}, "
        #                       f"变化率={throughput_change_rate:.2%}), "
        #                       f"梯度质量={grad_quality_score:.3f}, "
        #                       f"调整批次大小 {self.target_batch_size} -> {new_target}")
        #
        #         elif throughput_change_rate > 0.02:  # 吞吐量上升超过2%
        #             self.consecutive_improvements += 1
        #             self.consecutive_degradations = 0
        #
        #             # 性能改善：减小步长，精细调整
        #             if self.consecutive_improvements >= 3:
        #                 self.adjustment_step_size = max(self.adjustment_step_size * 0.9, self.min_step_size)
        #
        #             # 继续朝着改善的方向调整
        #             # 如果当前批次大小大于建议值，继续增大
        #             if self.target_batch_size > self.suggested_batch_size:
        #                 new_target = int(self.target_batch_size * self.adjustment_step_size)
        #             else:
        #                 new_target = int(self.target_batch_size / self.adjustment_step_size)
        #
        #             if sim_config.log_cluster_verbose:
        #                 print(f"[微观层微调] {self.name}: 吞吐量上升 "
        #                       f"({prev_avg_throughput:.2f} -> {recent_avg_throughput:.2f}, "
        #                       f"变化率={throughput_change_rate:.2%}), "
        #                       f"继续优化 {self.target_batch_size} -> {new_target}")
        #
        #         else:  # 性能稳定（-2% ~ +2%）
        #             # 根据梯度质量决定是否调整
        #             if recent_avg_grad_var > 1.5:  # 梯度方差较大，尝试增大批次
        #                 new_target = int(self.target_batch_size * 1.05)
        #                 if sim_config.log_cluster_verbose:
        #                     print(f"[微观层微调] {self.name}: 梯度方差大 ({recent_avg_grad_var:.3f}), "
        #                           f"增大批次 {self.target_batch_size} -> {new_target}")
        #             else:
        #                 # 性能稳定，保持当前配置或微调回建议值
        #                 new_target = int(0.9 * self.target_batch_size + 0.1 * self.suggested_batch_size)
        #                 if abs(new_target - self.target_batch_size) < 2:
        #                     new_target = self.target_batch_size  # 变化太小，不调整
        #
        #         # 确保在允许范围内
        #         new_target = max(min_bsz, min(new_target, max_bsz))
        #
        #         # 记录性能
        #         self.best_batch_size_performance[self.target_batch_size] = recent_avg_throughput
        #
        #         self.target_batch_size = new_target
        #     else:
        #         # 样本不足，保守调整
        #         new_target = int(0.8 * self.target_batch_size + 0.2 * self.suggested_batch_size)
        #         new_target = max(min_bsz, min(new_target, max_bsz))
        #         self.target_batch_size = new_target

        # 更新最后微调时间
        self.last_micro_adjustment_time = current_time

        current_throughput = self.recent_throughput_samples[-1] if self.recent_throughput_samples else 0.0

        # 记录实际执行的批次大小（用于上报给宏观层）
        self.actual_batch_size_history.append({
            'time': current_time,
            'batch_size': self.target_batch_size,
            'suggested': self.suggested_batch_size,
            'throughput': current_throughput
        })
        # 记录实际执行的批次大小
        # self.actual_batch_size_history.append({
        #     'time': current_time,
        #     'batch_size': self.target_batch_size,
        #     'suggested': self.suggested_batch_size,
        #     'throughput': self.recent_throughput_samples[-1] if self.recent_throughput_samples else 0,
        #     'grad_var': self.recent_gradient_variance_samples[-1] if self.recent_gradient_variance_samples else 0
        # })

    def collect_performance_metrics(self, throughput, grad_variance):
        """
        收集性能指标用于微调决策

        参数:
            throughput: 当前实际吞吐量（examples/sec）
            grad_variance: 当前梯度方差
        """
        self.recent_throughput_samples.append(throughput)
        self.recent_gradient_variance_samples.append(grad_variance)

        # 只保留最近的N个样本，避免内存占用过大
        max_samples = 10
        if len(self.recent_throughput_samples) > max_samples:
            self.recent_throughput_samples = self.recent_throughput_samples[-max_samples:]
        if len(self.recent_gradient_variance_samples) > max_samples:
            self.recent_gradient_variance_samples = self.recent_gradient_variance_samples[-max_samples:]

            # ===== 新增：追踪当前周期内的最优批次大小 =====
            # 如果当前吞吐量优于历史最优，更新记录
        if throughput > self.best_throughput_in_period:
            self.best_throughput_in_period = throughput
            self.best_bsz_in_period = self.target_batch_size
            self.best_bsz_timestamp = self.current_time

            if sim_config.log_cluster_verbose:
                print(f"[最优追踪] {self.name}: 发现更优批次大小 "
                      f"bsz={self.best_bsz_in_period}, "
                      f"throughput={throughput:.2f}")

    def get_batch_size_statistics(self):
        """
        获取当前调度周期内批次大小的统计信息
        返回给宏观层用于有效吞吐量模型的修正

        返回:
            dict: 包含最小值、最大值、均值、标准差、建议值、偏差和样本数
                  如果没有数据则返回 None
        """
        if not self.actual_batch_size_history:
            return None

        batch_sizes = [entry['batch_size'] for entry in self.actual_batch_size_history]

        # ======= 新增：提取吞吐量历史 =======
        throughputs = [entry['throughput'] for entry in self.actual_batch_size_history if 'throughput' in entry]

        # statistics = {
        #     'min': min(batch_sizes),
        #     'max': max(batch_sizes),
        #     'mean': np.mean(batch_sizes),
        #     'std': np.std(batch_sizes),
        #     'suggested': self.suggested_batch_size,
        #     'deviation': np.mean([abs(bs - self.suggested_batch_size)
        #                           for bs in batch_sizes]) if self.suggested_batch_size else 0,
        #     'samples': len(batch_sizes),
        #     'history': self.actual_batch_size_history.copy()  # 完整历史记录
        # }

        # 计算当前资源配置
        current_num_replicas = sum(self.placement) if self.placement else 0

        # ======= 新增：计算吞吐量变化率 (首尾相较) =======
        throughput_change_rate = 0.0
        if len(throughputs) > 1 and throughputs[0] > 0:
            throughput_change_rate = (throughputs[-1] - throughputs[0]) / throughputs[0]

        statistics = {
            'min': min(batch_sizes),
            'max': max(batch_sizes),
            'mean': np.mean(batch_sizes),
            'std': np.std(batch_sizes),
            'suggested': self.suggested_batch_size,
            'deviation': np.mean([abs(bs - self.suggested_batch_size)
                                  for bs in batch_sizes]) if self.suggested_batch_size else 0,
            'samples': len(batch_sizes),
            'history': self.actual_batch_size_history.copy(),
            # ===== 新增：最优批次大小信息 =====
            'best_bsz': self.best_bsz_in_period,
            'best_throughput': self.best_throughput_in_period,
            'best_bsz_timestamp': self.best_bsz_timestamp,
            'num_replicas': current_num_replicas,
            'cluster': self.current_cluster,
            # ======= 新增：用于论证的吞吐量衰减指标 =======
            'throughput_change_rate': throughput_change_rate
        }

        return statistics

    def reset_scheduling_period(self, current_time):
        """
        开始新的宏观调度周期时调用
        清空当前周期的历史数据，准备接收新的建议

        参数:
            current_time: 新调度周期开始的时间
        """
        # ===== 新增：保存本周期的最优批次大小到历史记录 =====
        if self.best_bsz_in_period is not None and self.best_throughput_in_period > 0:
            num_replicas = sum(self.placement) if self.placement else 0
            self.optimal_bsz_history.append({
                'timestamp': self.best_bsz_timestamp,
                'batch_size': self.best_bsz_in_period,
                'throughput': self.best_throughput_in_period,
                'num_replicas': num_replicas,
                'cluster': self.current_cluster,
                'period_start': self.current_scheduling_period_start,
                'period_end': current_time,
            })

            # 只保留最近的N个周期记录
            max_history = 10
            if len(self.optimal_bsz_history) > max_history:
                self.optimal_bsz_history = self.optimal_bsz_history[-max_history:]

            if sim_config.log_cluster_verbose:
                print(f"[周期总结] {self.name}: 本周期最优 bsz={self.best_bsz_in_period}, "
                      f"throughput={self.best_throughput_in_period:.2f}, "
                      f"num_replicas={num_replicas}")

        # 重置周期内的统计数据
        self.current_scheduling_period_start = current_time
        self.actual_batch_size_history = []
        self.best_bsz_in_period = None
        self.best_throughput_in_period = 0.0
        self.best_bsz_timestamp = 0

        if sim_config.log_cluster_verbose:
            print(f"[微观层] {self.name}: 重置调度周期, 时间={current_time}")

    def step(self, seconds, interference=0.0):
        if self.completion_time is not None:
            return
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return

        # ===== 新增：微观层自适应微调 =====
        # 在每次step开始时，检查是否需要进行批次大小微调
        self.micro_adjust_batch_size(self.current_time)

        # job does not use GPUs for `delay` seconds (checkpoint+restore)
        delay = min(self.rescale_time, seconds)
        # step delay time
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.wasted_gpu_seconds += delay * sum(self.placement)

        # update rescale time
        self.rescale_time -= delay

        # simulate training for any remaining seconds
        seconds -= delay
        if seconds > 0:
            self.used_gpu_seconds += (seconds * sum(self.placement))
        while seconds > 0 and self.completion_time is None:
            assert self.current_cluster is not None, "stepping on job without current_cluster set"
            application = self.applications[self.current_cluster]
            assert self.epoch < application.max_epochs
            # print(f"job: {self.name}, placement: {self.placement}")
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            num_nodes, num_gpus = len(placement), sum(placement)
            local_bsz = self.atomic_bsz * (self.accum_steps + 1)
            # one PMP replica can span many stages
            if self.num_stages > 1:
                num_replicas = num_gpus // self.num_stages
            else:
                num_replicas = num_gpus
            # compute minibatch size
            batch_size = num_replicas * local_bsz

            scale = batch_size / application.init_batch_size

            # Calculate true (simulated) efficiency.
            grad_sqr, grad_var = application.get_grad_stats(
                batch_size, self.epoch)
            gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)

            # Calculate true (simulated) throughput.
            # query xput with atomic_bsz (bsz per pipeline replica)
            query_bsz = self.atomic_bsz
            # check if job is PMP
            if self.num_stages > 1:
                # query xput with local_bsz (bsz per pipeline replica)
                query_bsz = local_bsz
            # get throughput for current placement with query_bsz per replica
            step_time, sync_time = application.get_throughput(
                placement, query_bsz)
            # compute time per accumulation step
            accum_time = step_time - sync_time
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, query_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            # do not count accum steps for PMP, only for DP jobs
            accum_steps = self.accum_steps if self.num_stages == 1 else 0

            # compute time per iter
            total_time = step_time + accum_time * accum_steps
            goodput = gain / total_time * (1.0 - interference)


            # 压力测试 1: Throughput Noise
            if getattr(sim_config, 'enable_throughput_noise', False):

                current_sens = getattr(self, 'current_sensitivity', 1.0)

                # 基础抖动标准差设为较小值 (例如 0.1)
                base_noise_std = getattr(sim_config, 'throughput_noise_std', 0.10)

                # 动态方差：敏感度越高，波动的幅度越大。乘法比加法更能放大敏感度的影响
                dynamic_noise_std = base_noise_std * (1.0 + current_sens)

                # 模拟真实系统：网络抖动只会拖慢进度，极少能加速。
                # 因此生成正态分布时，限制向上的最高收益 (最多加速 5%)，完全放开向下的惩罚
                noise_factor = np.random.normal(1.0, dynamic_noise_std)
                noise_factor = max(0.1, min(1.05, noise_factor)) # 最多加速 5%，最差降到 10%

                goodput *= noise_factor

            # 压力测试 2: Performance Cliffs
            if getattr(sim_config, 'enable_performance_cliffs', False):

                base_node_cliff_prob = getattr(sim_config, 'cliff_probability', 0.05)

                # 核心机制：分配的机器越多，至少有一台出问题的概率呈指数上升！
                # 公式：1 - (1 - 单节点安全率) ^ 节点数
                # 对比：8节点(Baseline)的触发率约为 33.6%，2节点(Risk-Adjusted)约为 9.7%
                actual_cliff_prob = 1.0 - (1.0 - base_node_cliff_prob) ** num_nodes

                # 掷骰子决定是否踩入悬崖
                if random.random() < actual_cliff_prob:
                    # 触发悬崖时，惩罚拉满 (吞吐量暴跌)
                    cliff_factor = getattr(sim_config, 'cliff_degradation_factor', 0.2)
                    goodput *= cliff_factor

                    # 可选：开启详细日志，观察 Baseline 如何被疯狂打断
                    if getattr(sim_config, 'log_exploration_verbose', False):
                        print(f"[Stress Test Cliff] {self.name} dropped to {cliff_factor*100}%. (Nodes: {num_nodes}, Prob: {actual_cliff_prob:.2f})")

            # goodput multiplier
            # goodput = self.multiplier * goodput
            # slowdown for job
            goodput = goodput * self.calibration_factor

            # ===== 新增：收集性能指标用于微观层决策 =====
            # 计算实际吞吐量（examples/sec）
            actual_throughput = batch_size / total_time if total_time > 0 else 0
            self.collect_performance_metrics(actual_throughput, grad_var)

            if sim_config.enable_risk_adjustment and self.current_cluster is not None:
                try:
                    goodput_fn = self.get_goodput_fn(self.current_cluster)
                    if goodput_fn is not None:
                        # 计算当前配置的敏感度
                        if sim_config.sensitivity_calculation_method == 'gradient':
                            sensitivity = goodput_fn.compute_sensitivity(
                                num_nodes, num_replicas, batch_size,
                                delta=sim_config.batch_size_perturbation
                            )
                        else:  # 'range'
                            sensitivity = goodput_fn.compute_sensitivity_range(
                                num_nodes, num_replicas, batch_size,
                                range_percent=sim_config.batch_size_fluctuation_range
                            )

                        # 转换为标量
                        self.current_sensitivity = float(sensitivity) if np.isscalar(sensitivity) else float(
                            sensitivity[0])

                        # # 计算风险调整后的goodput
                        # alpha = sim_config.risk_aversion_coefficient
                        # self.risk_adjusted_goodput = goodput - alpha * self.current_sensitivity

                        # 注意：此处的 current_sensitivity 可能是原始梯度，为了防止异常可以简单限幅
                        # ---- 修改后 ----
                        alpha = sim_config.risk_aversion_coefficient  # γ
                        beta = getattr(sim_config, 'risk_retention_factor', 0.9)  # β
                        logging_sensitivity = min(1.0, self.current_sensitivity)  # δ 在 compute_sensitivity 内部使用
                        # 风险惩罚公式: G' = G * max(β, 1 - γ * S)
                        # β 作为下界，确保不会被过度惩罚
                        self.risk_adjusted_goodput = goodput * max(beta, 1.0 - alpha * logging_sensitivity)

                        # 记录历史数据
                        self.batch_size_sensitivity_history.append({
                            'time': self.current_time,
                            'batch_size': batch_size,
                            'num_nodes': num_nodes,
                            'num_replicas': num_replicas,
                            'atomic_bsz': self.atomic_bsz,
                            'accum_steps': self.accum_steps,
                            'sensitivity': self.current_sensitivity,
                            'goodput_original': goodput,
                            'goodput_risk_adjusted': self.risk_adjusted_goodput,
                            'throughput': actual_throughput,
                            'grad_var': grad_var,
                        })

                        # 只保留最近50条记录
                        if len(self.batch_size_sensitivity_history) > 50:
                            self.batch_size_sensitivity_history = self.batch_size_sensitivity_history[-50:]

                        # 详细日志（可选）
                        if sim_config.log_risk_adjustment and self.current_time % 600 == 0:  # 每600秒打印一次
                            print(f"[风险监控] {self.name}: "
                                  f"bsz={batch_size}, "
                                  f"sensitivity={self.current_sensitivity:.4f}, "
                                  f"goodput={goodput:.2f} -> {self.risk_adjusted_goodput:.2f} "
                                  f"(降低{(goodput - self.risk_adjusted_goodput) / goodput * 100:.1f}%)")

                except Exception as e:
                    if sim_config.log_risk_adjustment:
                        print(f"[敏感度记录警告] {self.name}: {e}")
                    # 出错时使用默认值
                    self.current_sensitivity = 0.0
                    self.risk_adjusted_goodput = goodput
            else:
                # 未启用风险调整时的默认值
                self.current_sensitivity = 0.0
                self.risk_adjusted_goodput = goodput

            # Update current epoch and progress.
            next_progress = application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                self.execution_time += seconds
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                # update epoch duration
                assert len(self.epoch_duration) == self.epoch
                duration = round(float((application.get_progress(
                    self.epoch + 1) - application.get_progress(self.epoch)) / goodput))
                self.epoch_duration.append(duration)
                if self.epoch == application.max_epochs:
                    print(
                        f"Epoch durations: {self.name} -> {self.epoch_durations}")
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                completion_epoch = application.get_completion_epoch(batch_size)
                self.completion_epoch = completion_epoch
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = application.get_best_metric(
                    batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                self.execution_time += delta
                seconds -= delta
                # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement)
        self.current_time += seconds  # Add any remaining time.

    def reallocate(self, placement):
        old_placement, new_placement = self.placement, tuple(placement)
        if placement:
            if old_placement != new_placement:
                # print(f"RESCALE: job: {self.name}, cluster: {self.current_cluster}, placement: {old_placement} -> {new_placement}")

                # Update placement, num_stages, batch size per replica
                self.num_stages = self.applications[self.current_cluster].num_stages
                self.placement = new_placement
                self.update_local_bsz(self.placement)

                # Start startup/re-scale countdown
                self.rescale_time = self.applications[self.current_cluster].rescale_time or 30
                if self.num_restarts is None:
                    # starting for the first time
                    self.num_restarts = 0
                else:
                    # restarting
                    self.num_restarts += 1
        else:
            # De-allocate all resources.
            # print(f"SUSPEND: job: {self.name}")
            self.placement = ()
            self.atomic_bsz = 0

        # Record a rescale event
        new_rescale_event = (self.name, self.current_time, self.current_cluster, old_placement, new_placement,
                             self.rescale_time)
        self.rescale_events.append(new_rescale_event)
        return new_rescale_event

    def migrate(self, new_cluster, new_placement):
        # set current cluster
        prev_cluster = self.current_cluster
        # print(f"MIGRATE:: {self.name}, cluster: {prev_cluster} -> {new_cluster}")
        # update current cluster
        self.current_cluster = new_cluster
        if new_placement:
            self.placement = tuple(new_placement)
            self.update_local_bsz(self.placement)
            # Start startup/re-scale countdown.
            self.rescale_time = self.applications[self.current_cluster].rescale_time or 30
            # print(f"RESCALE_TIME: {self.name} --> {new_cluster}, {self.rescale_time}")
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
            # get num stages for this GPU type
            self.num_stages = self.applications[self.current_cluster].num_stages
        else:
            print(f"SUSPEND: job: {self.name}")
            # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0
        # Record a migrate event
        new_migrate_event = (self.name, self.current_time, prev_cluster, new_cluster, new_placement, self.rescale_time)
        self.migrate_events.append(new_migrate_event)
        return new_migrate_event

    def get_optimal_bsz_for_goodput(self):
        """
        获取用于goodput模块优化的最优批次大小信息

        返回:
            dict or None: 包含最优批次大小及其上下文信息，如果没有数据则返回None
        """
        if not self.optimal_bsz_history:
            return None

        # 返回最近一次的最优批次大小记录
        latest_optimal = self.optimal_bsz_history[-1]

        # 如果有多个周期的数据，计算平均最优批次大小
        if len(self.optimal_bsz_history) >= 3:
            recent_optimal_bszs = [entry['batch_size']
                                   for entry in self.optimal_bsz_history[-3:]]
            avg_optimal_bsz = int(np.mean(recent_optimal_bszs))
        else:
            avg_optimal_bsz = latest_optimal['batch_size']

        return {
            'latest_optimal_bsz': latest_optimal['batch_size'],
            'latest_throughput': latest_optimal['throughput'],
            'avg_optimal_bsz': avg_optimal_bsz,
            'num_replicas': latest_optimal['num_replicas'],
            'cluster': latest_optimal['cluster'],
            'history_length': len(self.optimal_bsz_history),
        }