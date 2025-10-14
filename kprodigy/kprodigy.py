"""
KProdigy optimizer - Enhanced Prodigy with performance optimizations and bug fixes.

Based on "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
by Konstantin Mishchenko and Aaron Defazio
https://arxiv.org/abs/2306.06101

Enhancements:
- Multi-tensor operations for ~21% GPU speedup
- Independent d estimation per parameter group for multi-component models
- Adaptive bias correction for improved stability
- Fixed EMA scaling and memory format handling
"""

import math
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

import torch
import torch.optim
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any

logger = logging.getLogger(__name__)


class KProdigy(torch.optim.Optimizer):
    r"""
    KProdigy: Enhanced Prodigy optimizer with performance improvements.
    
    Implements adaptive learning rate optimization with automatic D-adaptation.
    Typically works best with lr=1.0 (no manual tuning required).
   
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate multiplier. Default: 1.0
        betas (Tuple[float, float]):
            Coefficients for gradient and squared gradient moving averages.
            Default: (0.9, 0.999)
        beta3 (Optional[float]):
            Coefficient for D-adaptation. If None, uses sqrt(beta2).
            Default: None
        eps (float):
            Term added to denominator for numerical stability.
            Default: 1e-8
        weight_decay (float):
            Weight decay (L2 penalty).
            Default: 0.0
        decouple (bool):
            Use AdamW-style decoupled weight decay.
            Default: True
        use_bias_correction (bool):
            Enable Adam-style bias correction.
            Default: False
        safeguard_warmup (bool):
            Remove lr from D-estimate denominator during warmup.
            Default: False
        d0 (float):
            Initial D estimate for D-adaptation.
            Default: 1e-6
        d_coef (float):
            Coefficient in D estimate expression. Preferred tuning parameter.
            Default: 1.0
        growth_rate (float):
            Maximum multiplicative growth rate for D estimate.
            Default: float('inf')
        fsdp_in_use (bool):
            Set to True when using sharded parameters (FSDP).
            Default: False
        slice_p (int):
            Memory optimization: compute stats on every pth entry.
            Values ~11 are reasonable for memory-constrained scenarios.
            Default: 1
        foreach (bool):
            Enable multi-tensor operations for GPU speedup (~21% faster).
            Auto-falls back to CPU if needed.
            Default: True
        independent_d (bool):
            Calculate separate D estimates per parameter group.
            Essential for multi-component models (e.g., SDXL with UNet + Text Encoders).
            Default: False
    """
    
    def __init__(
        self,
        params: _params_t,
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        beta3: Optional[float] = None,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decouple: bool = True,
        use_bias_correction: bool = False,
        safeguard_warmup: bool = False,
        d0: float = 1e-6,
        d_coef: float = 1.0,
        growth_rate: float = float('inf'),
        fsdp_in_use: bool = False,
        slice_p: int = 1,
        foreach: bool = True,
        independent_d: bool = False
    ):
        if not 0.0 < d0:
            raise ValueError(f"Invalid d0 value: {d0}")
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not slice_p >= 1:
            raise ValueError(f"Invalid slice_p value: {slice_p}, must be >= 1")

        if decouple and weight_decay > 0:
            logger.info("Using decoupled weight decay")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            d0=d0,
            d_max=d0,
            d_numerator=0.0,
            d_coef=d_coef,
            k=0,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            safeguard_warmup=safeguard_warmup,
            fsdp_in_use=fsdp_in_use,
            slice_p=slice_p,
            foreach=foreach,
            independent_d=independent_d
        )
        
        self.d0 = d0
        super().__init__(params, defaults)
        
        self._detect_fsdp()

    def _detect_fsdp(self):
        """Detect FSDP usage during initialization."""
        for group in self.param_groups:
            if not group['fsdp_in_use']:
                for p in group['params']:
                    if hasattr(p, "_fsdp_flattened"):
                        group['fsdp_in_use'] = True
                        logger.info("FSDP detected, enabling distributed communication")
                        break

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        group = self.param_groups[0]
        independent_d = group.get('independent_d', False)
        
        if independent_d and len(self.param_groups) > 1:
            return self._step_independent_d(closure, loss)
        
        foreach = group.get('foreach', True)
        
        if foreach:
            can_use_foreach = all(
                p.is_cuda 
                for grp in self.param_groups 
                for p in grp['params'] 
                if p.grad is not None
            )
            if not can_use_foreach:
                foreach = False

        if foreach:
            return self._step_foreach(closure, loss)
        else:
            return self._step_single_tensor(closure, loss)

    def _step_single_tensor(self, closure, loss):
        """Single-tensor implementation (CPU fallback)."""
        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        d0 = group['d0']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
            # Adaptive bias correction for improved stability
            scale_factor = min(max(d / d0, 0.1), 1.0)
            bias_correction *= scale_factor
        else:
            bias_correction = 1.0

        dlr = d * lr * bias_correction
       
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3
        delta_numerator = 0.0

        # First pass: compute D estimate and update EMA
        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    "Setting different lr values in different parameter "
                    "groups is only supported for values of 0"
                )

            for p in group['params']:
                if p.grad is None:
                    continue
               
                grad = p.grad.data
               
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    
                    state['s'] = torch.zeros_like(
                        p.data.flatten()[::slice_p],
                        memory_format=torch.preserve_format
                    ).detach()

                    if p.data.flatten()[::slice_p].norm() > 0:
                        state['p0'] = p.data.flatten()[::slice_p].detach().clone()
                    else:
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format
                        ).detach()
                    
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format
                    ).detach()

                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p]
                    sliced_p = p.data.flatten()[::slice_p]
                    
                    delta_numerator += (d / d0) * dlr * torch.dot(
                        sliced_grad,
                        p0.data - sliced_p
                    ).item()

                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                    
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    
                    d_denom += s.abs().sum().item()

        # Compute new D estimate
        d_hat = d

        if d_denom == 0 and not fsdp_in_use:
            return loss
       
        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2, device='cuda')
                dist_tensor[0] = delta_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = d_numerator + dist_tensor[0].item()
                global_d_denom = dist_tensor[1].item()
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            
            if d == group['d0']:
                d = max(d, d_hat)
            
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

        # Second pass: apply parameter updates
        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = torch.maximum(
                    exp_avg_sq.sqrt(),
                    torch.tensor(eps, dtype=exp_avg_sq.dtype, device=exp_avg_sq.device)
                )

                if decay != 0 and decouple:
                    p.data.mul_(1 - decay * dlr)

                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr)

            group['k'] = k + 1

        return loss

    def get_d(self) -> float:
        """Get the current D estimate (distance to solution)."""
        return self.param_groups[0].get('d', self.d0)
    
    def _step_foreach(self, closure, loss):
        """Optimized multi-tensor implementation using foreach operations."""
        d_denom = 0.0

        group = self.param_groups[0]
        use_bias_correction = group['use_bias_correction']
        beta1, beta2 = group['betas']
        beta3 = group['beta3']
        if beta3 is None:
            beta3 = math.sqrt(beta2)
        k = group['k']

        d = group['d']
        d_max = group['d_max']
        d_coef = group['d_coef']
        d0 = group['d0']
        lr = max(group['lr'] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
            scale_factor = min(max(d / d0, 0.1), 1.0)
            bias_correction *= scale_factor
        else:
            bias_correction = 1.0

        dlr = d * lr * bias_correction
       
        growth_rate = group['growth_rate']
        decouple = group['decouple']
        fsdp_in_use = group['fsdp_in_use']

        d_numerator = group['d_numerator']
        d_numerator *= beta3
        delta_numerator = 0.0

        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']
            group_lr = group['lr']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    "Setting different lr values in different parameter "
                    "groups is only supported for values of 0"
                )

            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            states_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
               
                params_with_grad.append(p)
                grads.append(p.grad.data)
                
                grad = p.grad.data
               
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    
                    state['s'] = torch.zeros_like(
                        p.data.flatten()[::slice_p],
                        memory_format=torch.preserve_format
                    ).detach()

                    if p.data.flatten()[::slice_p].norm() > 0:
                        state['p0'] = p.data.flatten()[::slice_p].detach().clone()
                    else:
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format
                        ).detach()
                    
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format
                    ).detach()

                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                p0 = state['p0']

                exp_avg_sqs.append(exp_avg_sq)
                if beta1 > 0:
                    exp_avgs.append(state['exp_avg'])
                states_list.append(state)

                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p]
                    sliced_p = p.data.flatten()[::slice_p]
                    
                    delta_numerator += (d / d0) * dlr * torch.dot(
                        sliced_grad,
                        p0.data - sliced_p
                    ).item()

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    
                    d_denom += s.abs().sum().item()

            # Multi-tensor EMA updates
            if len(grads) > 0:
                if beta1 > 0 and len(exp_avgs) > 0:
                    torch._foreach_mul_(exp_avgs, beta1)
                    torch._foreach_add_(exp_avgs, grads, alpha=(1 - beta1))
                
                torch._foreach_mul_(exp_avg_sqs, beta2)
                torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=(1 - beta2))

        d_hat = d

        if d_denom == 0 and not fsdp_in_use:
            return loss
       
        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2, device='cuda')
                dist_tensor[0] = delta_numerator
                dist_tensor[1] = d_denom
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_d_numerator = d_numerator + dist_tensor[0].item()
                global_d_denom = dist_tensor[1].item()
            else:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

            d_hat = d_coef * global_d_numerator / global_d_denom
            
            if d == group['d0']:
                d = max(d, d_hat)
            
            d_max = max(d_max, d_hat)
            d = min(d_max, d * growth_rate)

        for group in self.param_groups:
            group['d_numerator'] = global_d_numerator
            group['d_denom'] = global_d_denom
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

        # Multi-tensor parameter updates
        for group in self.param_groups:
            decay = group['weight_decay']
            k = group['k']
            eps = group['eps']

            params_to_update = []
            exp_avgs_update = []
            exp_avg_sqs_update = []
            grads_update = []

            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                state['step'] += 1

                params_to_update.append(p.data)
                exp_avg_sqs_update.append(state['exp_avg_sq'])
                grads_update.append(p.grad.data)
                if beta1 > 0:
                    exp_avgs_update.append(state['exp_avg'])

            if len(params_to_update) > 0:
                # Batch sqrt operation
                sqrts = torch._foreach_sqrt(exp_avg_sqs_update)
                
                eps_tensor = torch.tensor(eps, device=sqrts[0].device, dtype=sqrts[0].dtype)
                denoms = [torch.maximum(sqrt, eps_tensor) for sqrt in sqrts]

                if decay != 0 and decouple:
                    torch._foreach_mul_(params_to_update, 1 - decay * dlr)

                if beta1 > 0:
                    torch._foreach_addcdiv_(params_to_update, exp_avgs_update, denoms, value=-dlr)
                else:
                    torch._foreach_addcdiv_(params_to_update, grads_update, denoms, value=-dlr)

            group['k'] = k + 1

        return loss
    
    def _step_independent_d(self, closure, loss):
        """Multi-group implementation with independent D estimation per group.
        
        Essential for multi-component models (e.g., SDXL with UNet + Text Encoders)
        to prevent "burning" sensitive components with mismatched learning rates.
        """
        for group in self.param_groups:
            use_bias_correction = group['use_bias_correction']
            beta1, beta2 = group['betas']
            beta3 = group['beta3']
            if beta3 is None:
                beta3 = math.sqrt(beta2)
            
            k = group['k']
            d = group['d']
            d_max = group['d_max']
            d_coef = group['d_coef']
            group_lr = group['lr']
            growth_rate = group['growth_rate']
            decouple = group['decouple']
            eps = group['eps']
            d0 = group['d0']
            safeguard_warmup = group['safeguard_warmup']
            slice_p = group['slice_p']
            decay = group['weight_decay']
            
            if use_bias_correction:
                bias_correction = ((1 - beta2**(k+1))**0.5) / (1 - beta1**(k+1))
                scale_factor = min(max(d / d0, 0.1), 1.0)
                bias_correction *= scale_factor
            else:
                bias_correction = 1.0

            dlr = d * group_lr * bias_correction
            
            d_numerator = group['d_numerator']
            d_numerator *= beta3
            delta_numerator = 0.0
            d_denom = 0.0
            
            s_tensors = []
            
            for p in group['params']:
                if p.grad is None:
                    continue
               
                grad = p.grad.data
               
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    
                    state['s'] = torch.zeros_like(
                        p.data.flatten()[::slice_p],
                        memory_format=torch.preserve_format
                    ).detach()

                    if p.data.flatten()[::slice_p].norm() > 0:
                        state['p0'] = p.data.flatten()[::slice_p].detach().clone()
                    else:
                        state['p0'] = torch.tensor(0, device=p.device, dtype=p.dtype)

                    if beta1 > 0:
                        state['exp_avg'] = torch.zeros_like(
                            p.data,
                            memory_format=torch.preserve_format
                        ).detach()
                    
                    state['exp_avg_sq'] = torch.zeros_like(
                        p.data,
                        memory_format=torch.preserve_format
                    ).detach()

                exp_avg_sq = state['exp_avg_sq']
                s = state['s']
                p0 = state['p0']

                if group_lr > 0.0:
                    sliced_grad = grad.flatten()[::slice_p]
                    sliced_p = p.data.flatten()[::slice_p]
                    
                    delta_numerator += (d / d0) * dlr * torch.dot(
                        sliced_grad,
                        p0.data - sliced_p
                    ).item()

                    if beta1 > 0:
                        exp_avg = state['exp_avg']
                        exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                    
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                    if safeguard_warmup:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * d))
                    else:
                        s.mul_(beta3).add_(sliced_grad, alpha=((d / d0) * dlr))
                    
                    s_tensors.append(s)

            # Batch operation for D denominator
            if s_tensors:
                d_denom = torch.cat(s_tensors).abs().sum().item()
            else:
                d_denom = 0.0

            d_hat = d

            if d_denom > 0 and group_lr > 0.0:
                global_d_numerator = d_numerator + delta_numerator
                global_d_denom = d_denom

                d_hat = d_coef * global_d_numerator / global_d_denom
                
                if d == d0:
                    d = max(d, d_hat)
                
                d_max = max(d_max, d_hat)
                d = min(d_max, d * growth_rate)

            group['d_numerator'] = global_d_numerator if d_denom > 0 else d_numerator
            group['d_denom'] = global_d_denom if d_denom > 0 else 0.0
            group['d'] = d
            group['d_max'] = d_max
            group['d_hat'] = d_hat

            dlr = d * group_lr * bias_correction

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                state = self.state[p]
                exp_avg_sq = state['exp_avg_sq']

                state['step'] += 1

                denom = torch.maximum(
                    exp_avg_sq.sqrt(),
                    torch.tensor(eps, dtype=exp_avg_sq.dtype, device=exp_avg_sq.device)
                )

                if decay != 0 and decouple:
                    p.data.mul_(1 - decay * dlr)

                if beta1 > 0:
                    exp_avg = state['exp_avg']
                    p.data.addcdiv_(exp_avg, denom, value=-dlr)
                else:
                    p.data.addcdiv_(grad, denom, value=-dlr)

            group['k'] = k + 1

        return loss
