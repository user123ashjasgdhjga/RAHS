# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import autograd
import autograd.numpy as anp
import numpy as np
import collections
import scipy.optimize
import scipy.stats


# Parameters for a performance model which predicts the per-step time of
# distributed SGD using all-reduce. At a high level, models compute time and
# network time separately, and combines them with some degree of overlap.
# Compute time is modeled as a linear function of the local batch size.
# Network time is modeled using different parameters depending on if the job
# is inter-node (there exists a pair of replicas on different nodes), or
# intra-node (all replicas are on the same node). For both cases, network time
# is modeled as a constant term plus a retrogression term which increases
# linearly with the total number of replicas.
PerfParams = collections.namedtuple("PerfParams", [
    # T_compute ~ alpha_c + beta_c * local_bsz +
    #             (alpha_a + beta_a * local_bsz) * accumulation_steps
    "alpha_c",  # Constant term of compute time
    "beta_c",   # Multiplicative factor of compute time
    # If inter-node: T_network ~ alpha_n + beta_n * replicas
    "alpha_n",  # Constant term of inter-node network time
    "beta_n",   # Retrogression factor of inter-node network time
    # If intra-node: T_network ~ alpha_r + beta_r * replicas
    "alpha_r",  # Constant term of intra-node network time
    "beta_r",   # Retrogression factor of intra-node network time
    # T_step ~ (T_compute ^ gamma + T_network ^ gamma) ^ (1 / gamma)
    # Essentially is a p-norm where p = gamma. When p ~ 1 then
    # T_step ~ T_compute + T_network, indicating no overlap between compute
    # and network. When p -> infinity then T_step = max(T_compute, T_network),
    # indicating perfect overlap. We limit gamma to [1, 10] since 10 is close
    # enough to approximate the max function for our purposes.
    "gamma",    # Models the degree of overlap between compute and network
])

GradParams = collections.namedtuple("GradParams", ["sqr", "var"])

class GoodputFunction(object):
    def __init__(self, perf_params, grad_params, init_batch_size):
        self._perf_params = PerfParams(*perf_params)
        self._grad_params = GradParams(*grad_params)
        self._init_batch_size = init_batch_size

    def __call__(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        return self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)

    def evaluate(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        assert np.all(self._init_batch_size <= batch_size)
        return self.throughput(num_nodes, num_replicas, atomic_bsz,
                               accum_steps) * self.efficiency(batch_size)

    def throughput(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        accum_time = _predict_accum_time(self._perf_params, atomic_bsz)
        network_time = _predict_network_time(self._perf_params,
                                             num_nodes, num_replicas)
        optim_time = np.exp(_predict_log_optim_time(self._perf_params,
                                                    accum_time, network_time))
        total_time = accum_steps * accum_time + optim_time
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        return (batch_size / total_time)

    def efficiency(self, batch_size):
        grad_sqr = self._grad_params.sqr
        grad_var = self._grad_params.var
        scale = batch_size / self._init_batch_size
        denom = grad_var / scale + grad_sqr
        gain = np.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
        return gain / scale

    # IF tune_bsz==False, then max_batch_size MUST be target_batch_size
    def optimize(self, num_nodes, num_replicas, max_batch_size=None,
                 atomic_bsz_range=None, accumulation=False, tune_bsz=True):
        assert np.all(np.less_equal(1, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        if max_batch_size is None:
            max_batch_size = self._init_batch_size
        assert self._init_batch_size <= max_batch_size
        atomic_bsz_range = atomic_bsz_range or (None, None)
        min_atomic_bsz = atomic_bsz_range[0] or 1
        max_atomic_bsz = atomic_bsz_range[1] or max_batch_size
        # Remember what the output shape/format should be and flatten inputs.
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        output_scalar = np.isscalar(num_nodes) or np.isscalar(num_replicas)
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Samples 50 different total batch sizes in geometric space.
        if not tune_bsz:
            min_batch_size = np.asarray([max_batch_size])
        else:
            min_batch_size = np.maximum(self._init_batch_size,
                                        min_atomic_bsz * num_replicas)
        batch_size = np.geomspace(min_batch_size, max_batch_size)
        # print(f"goodput.optimize(): batch_size_range={np.min(batch_size), np.max(batch_size)}")
        local_bsz = batch_size / num_replicas
        eps = 1e-8  # Tolerance for floor/ceil operations.
        if accumulation:
            # If local_bsz size exceeds the max atomic batch size, split it
            # into a number of batches to form (atomic_bsz, accum_steps) such
            # that (atomic_bsz * (accum_steps + 1)) is close to local_bsz.
            #
            # If num_replicas == 1 and local_bsz > self._init_batch_size, then
            # set accum_steps to at least 1. This is because the gradient
            # statistics used for scaling up the learning rate are inaccurate
            # when there is only one atomic minibatch to estimate them from.
            accum_steps = np.ceil(local_bsz / max_atomic_bsz - eps) - 1
            accum_steps = np.where(
                np.logical_and(num_replicas == 1,
                               local_bsz > self._init_batch_size + eps),
                np.maximum(accum_steps, 1), accum_steps).astype(int)
        else:
            accum_steps = np.zeros_like(local_bsz, dtype=np.int)
        atomic_bsz = np.ceil(local_bsz / (accum_steps + 1) - eps).astype(int)
        # Evaluate the goodput of all candidate configurations.
        goodput = self.evaluate(num_nodes, num_replicas,
                                atomic_bsz, accum_steps)

        # Set the goodput of invalid configurations to 0.0.
        goodput = np.where((min_atomic_bsz <= atomic_bsz) &
                           (atomic_bsz <= max_atomic_bsz), goodput, 0.0)
        indices = np.argmax(goodput, axis=0), np.arange(goodput.shape[1])
        # Restore the correct output shape and return results.
        goodput = goodput[indices].reshape(output_shape)
        atomic_bsz = atomic_bsz[indices].reshape(output_shape)
        accum_steps = accum_steps[indices].reshape(output_shape)
        if output_scalar:
            goodput = goodput.item()
            atomic_bsz = atomic_bsz.item()
            accum_steps = accum_steps.item()
        return goodput, atomic_bsz, accum_steps

    def compute_sensitivity(self, num_nodes, num_replicas, batch_size, delta=0.05):
        """
        计算有效吞吐量对批次大小变化的敏感度（使用数值梯度）

        核心思想：
        S = |dG/db| ≈ |G(b+Δb) - G(b-Δb)| / (2*Δb)

        参数:
            num_nodes: 节点数
            num_replicas: 副本数
            batch_size: 当前批次大小
            delta: 扰动比例（默认±5%）

        返回:
            敏感度值（梯度的绝对值）
        """
        import numpy as np

        # 处理输入格式
        batch_size = np.atleast_1d(batch_size)
        num_nodes = np.atleast_1d(num_nodes)
        num_replicas = np.atleast_1d(num_replicas)

        # 确保batch_size是有效的
        valid_mask = batch_size > 0
        if not np.any(valid_mask):
            return np.zeros_like(batch_size, dtype=np.float32)

        sensitivity = np.zeros_like(batch_size, dtype=np.float32)

        # 只对有效配置计算敏感度
        valid_batch_size = batch_size[valid_mask]
        valid_num_nodes = num_nodes[valid_mask] if len(num_nodes) > 1 else num_nodes
        valid_num_replicas = num_replicas[valid_mask] if len(num_replicas) > 1 else num_replicas

        # 计算扰动后的批次大小
        delta_b = valid_batch_size * delta
        batch_size_plus = valid_batch_size + delta_b
        batch_size_minus = np.maximum(valid_batch_size - delta_b, self._init_batch_size)

        # 计算对应的配置
        def compute_config(bs, nr):
            local_bsz = bs / nr
            eps = 1e-8
            # 简化：假设max_local_bsz = init_batch_size * 2
            max_local_bsz = self._init_batch_size * 2
            accum_steps = np.maximum(np.ceil(local_bsz / max_local_bsz - eps) - 1, 0).astype(int)
            atomic_bsz = np.ceil(local_bsz / (accum_steps + 1) - eps).astype(int)
            atomic_bsz = np.maximum(atomic_bsz, 1)
            return atomic_bsz, accum_steps

        try:
            # 计算G(b + Δb)
            atomic_bsz_plus, accum_steps_plus = compute_config(batch_size_plus, valid_num_replicas)
            goodput_plus = self.evaluate(valid_num_nodes, valid_num_replicas,
                                         atomic_bsz_plus, accum_steps_plus)

            # 计算G(b - Δb)
            atomic_bsz_minus, accum_steps_minus = compute_config(batch_size_minus, valid_num_replicas)
            goodput_minus = self.evaluate(valid_num_nodes, valid_num_replicas,
                                          atomic_bsz_minus, accum_steps_minus)

            # 计算数值梯度: |dG/db| ≈ |G(b+Δb) - G(b-Δb)| / (2*Δb)
            valid_sensitivity = np.abs(goodput_plus - goodput_minus) / (2 * delta_b + 1e-10)
            sensitivity[valid_mask] = valid_sensitivity

        except Exception as e:
            print(f"Warning: Failed to compute sensitivity: {e}")
            sensitivity[valid_mask] = 0.0

        return sensitivity

    def compute_sensitivity_range(self, num_nodes, num_replicas, batch_size, range_percent=0.15):
        """
        计算有效吞吐量在批次大小±range范围内的波动性

        核心思想：
        S = max(G) - min(G) 在 [b*(1-r), b*(1+r)] 范围内

        参数:
            num_nodes: 节点数
            num_replicas: 副本数
            batch_size: 当前批次大小
            range_percent: 范围百分比（默认±15%）

        返回:
            敏感度值（吞吐量波动范围）
        """
        import numpy as np

        batch_size = np.atleast_1d(batch_size)
        num_nodes = np.atleast_1d(num_nodes)
        num_replicas = np.atleast_1d(num_replicas)

        valid_mask = batch_size > 0
        if not np.any(valid_mask):
            return np.zeros_like(batch_size, dtype=np.float32)

        sensitivity = np.zeros_like(batch_size, dtype=np.float32)

        valid_batch_size = batch_size[valid_mask]
        valid_num_nodes = num_nodes[valid_mask] if len(num_nodes) > 1 else num_nodes
        valid_num_replicas = num_replicas[valid_mask] if len(num_replicas) > 1 else num_replicas

        # 计算范围内的多个采样点
        min_bs = np.maximum(valid_batch_size * (1 - range_percent), self._init_batch_size)
        max_bs = valid_batch_size * (1 + range_percent)

        # 采样5个点
        try:
            goodputs = []
            for i in range(5):
                t = i / 4.0  # 0, 0.25, 0.5, 0.75, 1.0
                bs = min_bs + t * (max_bs - min_bs)

                local_bsz = bs / valid_num_replicas
                eps = 1e-8
                max_local_bsz = self._init_batch_size * 2
                accum_steps = np.maximum(np.ceil(local_bsz / max_local_bsz - eps) - 1, 0).astype(int)
                atomic_bsz = np.ceil(local_bsz / (accum_steps + 1) - eps).astype(int)
                atomic_bsz = np.maximum(atomic_bsz, 1)

                gp = self.evaluate(valid_num_nodes, valid_num_replicas, atomic_bsz, accum_steps)
                goodputs.append(gp)

            # 敏感度 = 最大值 - 最小值（波动范围）
            goodputs = np.array(goodputs)
            valid_sensitivity = np.max(goodputs, axis=0) - np.min(goodputs, axis=0)
            sensitivity[valid_mask] = valid_sensitivity

        except Exception as e:
            print(f"Warning: Failed to compute range sensitivity: {e}")
            sensitivity[valid_mask] = 0.0

        return sensitivity

    def compute_risk_adjusted_goodput(self, num_nodes, num_replicas, atomic_bsz, accum_steps,
                                      alpha=0.1, method='gradient'):
        """
        计算风险调整后的有效吞吐量

        公式: G'_ij = G_ij - α * S_ij

        参数:
            num_nodes, num_replicas, atomic_bsz, accum_steps: 配置参数
            alpha: 风险厌恶系数
            method: 敏感度计算方法 ('gradient' 或 'range')

        返回:
            风险调整后的goodput值
        """
        # 计算原始goodput
        goodput = self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)

        # 计算批次大小
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)

        # 计算敏感度
        if method == 'gradient':
            sensitivity = self.compute_sensitivity(num_nodes, num_replicas, batch_size)
        elif method == 'range':
            sensitivity = self.compute_sensitivity_range(num_nodes, num_replicas, batch_size)
        else:
            raise ValueError(f"Unknown sensitivity method: {method}")

        # 应用风险惩罚
        risk_adjusted_goodput = goodput - alpha * sensitivity

        return risk_adjusted_goodput, goodput, sensitivity

def fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                    accum_step_time, optim_step_time):
    # Fit the performance model given accum time and optim time measurements
    # for different configurations of num_nodes, num_replicas, and atomic_bsz.

    # HACK: We want to use the original numpy module for calls from the
    # SpeedupFunction for performance reasons, but also need those functions to
    # use autograd.numpy when we want to differentiate them. We patch the
    # global np reference only for the code invoked from this function.
    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    num_nodes = np.array(num_nodes)
    num_replicas = np.array(num_replicas)
    local_bsz = np.array(atomic_bsz)
    accum_step_time = np.array(accum_step_time)
    optim_step_time = np.array(optim_step_time)

    # Set initial params to reasonable values.
    params = [1e-1, 1e-2] * 3 + [1.0 + 1e-3]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    # bounds to avoid numerical instability issues.
    lower = [1e-8, 1e-8] * 3 + [1.0]
    upper = [np.inf, np.inf] * 3 + [10.0]
    if len(np.unique(atomic_bsz)) == 1:
        # Fix alpha_c if only observed a single atomic batch size.
        # This makes the speedup model optimistic with respect to
        # scaling up the batchsize. This will assign equal weight
        # to the constant and multplicative factors for accum time
        # if there is only a single datapoint (which is by far the
        # most likely case for this scenario)
        params[0] = upper[0] = lower[0] = np.mean(accum_step_time) / 2
    if not np.any(num_nodes > 1):
        # Fix alpha_n and beta_n if no multi-node observations.
        params[2] = upper[2] = lower[2]
        params[3] = upper[3] = lower[3]
    if not np.any(np.logical_and(num_nodes == 1, num_replicas > 1)):
        # Fix alpha_r and beta_r if no single-node/multi-replica observations.
        params[4] = upper[4] = lower[4]
        params[5] = upper[5] = lower[5]
    if not np.any(num_replicas > 2):
        # Fix beta_n and beta_r if no replicas > 2.
        params[3] = upper[3] = lower[3]
        params[5] = upper[5] = lower[5]
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True)
    args = (num_nodes, num_replicas, atomic_bsz,
            accum_step_time, optim_step_time)
    # FIXME: need to handle optimization failures and propagate to the Trainer.
    grad_fn = autograd.grad(_obj_fn)
    # changing tol to 1e-5 from machine precision for speed
    result = scipy.optimize.minimize(_obj_fn, params, args=args, jac=grad_fn, 
                                     bounds=bounds, tol=1e-5)
    params = result.x
    if not any(num_nodes > 1):
        # Enforce prior: alpha_n and beta_n are at least alpha_r and beta_r.
        params[2] = max(params[2], params[4] * 1.1)
        params[3] = max(params[3], params[5] * 1.1)
    np = orig_np  # Restore original numpy.
    return PerfParams(*params)

def _rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())

def _obj_fn(params, num_nodes, num_replicas, atomic_bsz, accum_step_time, optim_step_time):
    params = PerfParams(*params)
    pred_accum = _predict_accum_time(params, atomic_bsz)
    pred_network = _predict_network_time(params, num_nodes, num_replicas)
    pred_log_optim = _predict_log_optim_time(params, pred_accum, pred_network)
    # RMSLError of accum step time predictions.
    err1 = _rmse(np.log(pred_accum), np.log(accum_step_time))
    # RMSLError of optim step time predictions.
    err2 = _rmse(pred_log_optim, np.log(optim_step_time))
    # L2 regularization towards a smaller gamma, because it's easier to
    # optimize the alpha and beta parameters when gamma is smaller.
    reg1 = 1e-3 * (params.gamma - 1) ** 2
    # Penalize retrogression terms to prefer a more optimistic model.
    reg2 = 1e-2 * ((params.beta_n / params.alpha_n) ** 2 +
                   (params.beta_r / params.alpha_r) ** 2)
    return err1 + err2 + reg1 + reg2

def _predict_accum_time(params, atomic_bsz):
    params = PerfParams(*params)
    # Forward/backward passes should scale linearly with the batch size.
    return params.alpha_c + params.beta_c * atomic_bsz

def _predict_log_optim_time(params, accum_time, network_time):
    gamma = PerfParams(*params).gamma
    return np.log(accum_time ** gamma + network_time ** gamma) / gamma

def _predict_network_time(params, num_nodes, num_replicas):
    params = PerfParams(*params)
    # Select the most significant link between replicas, currently either
    # inter-node (nodes > 1) or intra-node (replicas > 1). Note that if
    # replicas == 1 then neither of these two conditions are matched.
    conds = [num_nodes > 1, num_replicas > 1]
    # Bandwidth is bottlenecked by the most significant link, alpha models
    # the overhead of transferring data across that link.
    bottleneck = np.select(conds, [params.alpha_n, params.alpha_r], 1e-8)
    # Assuming ring all-reduce, communication happens in a number of rounds
    # equal to the number of replicas. beta models the performance
    # retrogression from increasing the number of replicas beyond 2.
    retrogress = np.select(conds, [params.beta_n, params.beta_r], 1e-8)
    retrogress = retrogress * np.maximum(num_replicas - 2, 1e-8)
    return bottleneck + retrogress



class GoodputFunctionPMP(GoodputFunction):
    def __init__(self, perf_params, grad_params, init_batch_size, num_stages, num_micro):
        super().__init__(perf_params, grad_params, init_batch_size)
        self._num_stages = num_stages
        self._num_microbsz = num_micro
        # for numerical stability in optimization
        # (does not affect results since the rows are normalized to min-value 1)
        self._GOODPUT_MULTIPLER = 100

    # num_nodes         --> number of nodes
    # num_replicas      --> number of DP replicas of pipeline
    # micro_bsz         --> micro batch size
    # num_micro         --> number of micro batches
    def throughput(self, num_nodes, num_replicas, micro_bsz=None, num_micro=None):
        print(
            f"called throughput with {num_nodes}, {num_replicas}, {micro_bsz}, {num_micro}")
        micro_bsz = micro_bsz if micro_bsz else np.asarray(1)
        num_stages = np.asarray(self._num_stages)
        num_micro = num_micro if num_micro else np.asarray(self._num_microbsz)

        params = PerfParams(*self._perf_params)
        # compute the time for the forward pass for a micro batch
        fwd_stage = np.asarray(params.alpha_c + params.beta_c * micro_bsz)
        accum_time = np.asarray(
            3 * ((num_stages - 1) * fwd_stage + num_micro * fwd_stage))

        network_time = _predict_network_time(
            self._perf_params, num_nodes, num_replicas)
        total_time = accum_time + network_time
        batch_size = np.asarray(num_replicas * micro_bsz * num_micro)
        return (batch_size / total_time)

    # num_replicas      --> number of DP replicas of pipeline
    # num_nodes         --> number of nodes
    # max_batch_size    --> maximum batch size
    # atomic_bsz_range  --> ignored
    # accumulation      --> ignored (always False)
    # tune_bsz          --> ignored (always True)
    def optimize(self, num_nodes, num_replicas, max_batch_size=None,
                 atomic_bsz_range=None, accumulation=False, tune_bsz=True):
        print(
            f"Calling optimize with {num_nodes}, {num_replicas}, {max_batch_size}, {atomic_bsz_range}, {accumulation}, {tune_bsz}")
        output_scalar = np.isscalar(num_nodes) or np.isscalar(num_replicas)
        if output_scalar:
            num_nodes = np.array(num_nodes, dtype=np.float32)
            num_replicas = np.array(num_replicas, dtype=np.float32)

        # out_goodput --> goodput at each input config,
        # zero if num_replicas is a non-integer multiple of self._num_stages
        out_goodput = np.zeros_like(num_nodes, dtype=np.float32)
        out_atomic_bsz = np.zeros_like(num_nodes, dtype=np.int32)
        out_accum_steps = np.zeros_like(num_nodes, dtype=np.int32)

        # set out_atomic_bsz (microbsz) to 1 for valid configs
        out_atomic_bsz += 1
        out_accum_steps += self._num_microbsz - 1

        # set to 0 for configs with bsz > max_batch_size
        total_bsz = num_replicas * (out_atomic_bsz * (out_accum_steps + 1))
        if max_batch_size is not None:
            valid_configs = (total_bsz <= max_batch_size)
            # if non are valid, return early
            if not np.any(valid_configs):
                if output_scalar:
                    out_goodput = out_goodput.item()
                    out_atomic_bsz = out_atomic_bsz.item()
                    out_accum_steps = out_accum_steps.item()
                return out_goodput, out_atomic_bsz, out_accum_steps

        # query xput
        out_goodput = self.throughput(
            num_nodes, num_replicas) * self._GOODPUT_MULTIPLER

        # query and multiply efficiency for valid configs
        out_goodput *= self.efficiency(total_bsz)

        # set goodput, atomic_bsz, accum_steps to 0 for invalid configs
        if max_batch_size is not None and np.any(~valid_configs):
            # valid_configs would be computed already
            # if max_batch_size is not None
            out_goodput[~valid_configs] = 0
            out_atomic_bsz[~valid_configs] = 0
            out_accum_steps[~valid_configs] = 0
        if output_scalar:
            out_goodput = out_goodput.item()
            out_atomic_bsz = out_atomic_bsz.item()
            out_accum_steps = out_accum_steps.item()
        print(
            f"out_goodput: {out_goodput}, out_atomic_bsz: {out_atomic_bsz}, out_accum_steps: {out_accum_steps}")
        return out_goodput, out_atomic_bsz, out_accum_steps

    def compute_sensitivity(self, num_nodes, num_replicas, batch_size, delta=0.05):
        """
        PMP任务的敏感度计算（简化版）
        由于PMP任务的批次大小是固定的，敏感度主要来自副本数变化
        """
        import numpy as np
        # PMP任务通常批次大小固定，返回较小的敏感度
        return np.zeros_like(batch_size, dtype=np.float32) + 0.01

    def compute_sensitivity_range(self, num_nodes, num_replicas, batch_size, range_percent=0.15):
        """PMP任务的范围敏感度（简化版）"""
        import numpy as np
        return np.zeros_like(batch_size, dtype=np.float32) + 0.01