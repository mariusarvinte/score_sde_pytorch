# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
from scipy import integrate
from models import utils as mutils

from torchdiffeq import odeint_adjoint as odeint_adjoint
from torchdiffeq import odeint as odeint

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    # Force gradients even when under no_grad context
    with torch.enable_grad():
        x.requires_grad_(True)
        fn_eps = torch.sum(fn(x, t) * eps)
        grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    # x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


class ODEFunc(torch.nn.Module):
    def __init__(self, model, shape, drift_fn, div_fn, epsilon):
        super().__init__()
        self.shape    = shape
        self.model    = model
        self.drift_fn = drift_fn
        self.div_fn   = div_fn
        self.epsilon  = epsilon
        
    def forward(self, t, x):
        sample    = torch.reshape(x[:-self.shape[0]], self.shape).type(torch.float32)
        vec_t     = torch.ones(sample.shape[0], device=sample.device) * t
        drift     = self.drift_fn(self.model, sample, vec_t).reshape((-1,))
        logp_grad = self.div_fn(self.model, sample, vec_t, self.epsilon).reshape((-1,))
        
        return torch.concat([drift, logp_grad], dim=0)


class ODESolver(torch.nn.Module):
  def __init__(self, model, sde, inverse_scaler, timesteps=(1e-5, 1), rtol=1e-5, atol=1e-5, 
               method='rk4', adjoint_grads=True,
               eps=1e-5, ode_step_size=0.01, epsilon=None, shape=None):
      super().__init__()
      # Store objects
      self.model           = model
      self.method          = method
      self.sde, self.eps   = sde, eps
      self.timesteps       = torch.tensor(timesteps, device='cuda:0')
      self.rtol, self.atol = rtol, atol
      self.inverse_scaler  = inverse_scaler
      self.ode_step_size   = ode_step_size
      self.adjoint_grads   = adjoint_grads
      
      # Create a inner core module
      self.ode_func = ODEFunc(self.model, shape, self.drift_fn, 
                              self.div_fn, epsilon)
      
  def drift_fn(self, model, x, t):
    score_fn = mutils.get_score_fn(self.sde, model, train=False,
                                   continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = self.sde.reverse(score_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def div_fn(self, model, x, t, noise):
    return get_div_fn(lambda xx, tt: self.drift_fn(model, xx, tt))(x, t, noise)
  
  def forward(self, data):
    shape = data.shape
    init  = torch.concat([torch.flatten(data), 
                          torch.zeros(shape[0], device=data.device)], dim=0)
    
    if self.method == 'rk4' or self.method == 'midpoint' or self.method == 'euler':
        options = {'step_size': self.ode_step_size}
    elif self.method == 'bosh3' or self.method == 'adaptive_heun':
        options = None
    
    if self.adjoint_grads:
        solution = odeint_adjoint(self.ode_func, init, self.timesteps, options=options,
                          rtol=self.rtol, atol=self.atol, method=self.method)
    else:
        solution = odeint(self.ode_func, init, self.timesteps, options=options,
                          rtol=self.rtol, atol=self.atol, method=self.method)
    
    zp = solution[-1]
    z  = torch.reshape(zp[:-shape[0]], shape)
    delta_logp = torch.reshape(zp[-shape[0]:], (shape[0],))
    nfe = -1
          
    prior_logp    = self.sde.prior_logp(z)
    bpd           = -(prior_logp + delta_logp) / np.log(2)
    N             = np.prod(shape[1:])
    bpd           = bpd / N
    nll           = bpd
    
    return nll