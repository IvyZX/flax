# Copyright 2024 The Flax Authors.
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

import jax

from absl.testing import absltest
from flax import linen as nn
from flax import nnx
from flax.nnx import bridge
import numpy as np


class TestCompatibility(absltest.TestCase):
  def test_functional(self):
    # Functional API for NNX Modules
    functional = bridge.functional(nnx.Linear)(32, 64)
    state = functional.init(rngs=nnx.Rngs(0))
    x = jax.numpy.ones((1, 32))
    y, updates = functional.apply(state)(x)

  def test_linen_to_nnx(self):
    ## Wrapper API for Linen Modules
    linen_module = nn.Dense(features=64)
    x = jax.numpy.ones((1, 32))
    module = bridge.LinenToNNX(linen_module, rngs=nnx.Rngs(0))  # init
    y = module(x)  # apply
    assert y.shape == (1, 64)
  
  def test_linen_to_nnx_submodule(self):
    class NNXOuter(nnx.Module):
      def __init__(self, dout: int, *, rngs: nnx.Rngs):
        self.nn_dense1 = bridge.LinenToNNX(nn.Dense(dout, use_bias=False), rngs=rngs)
        self.b = nnx.Param(jax.random.uniform(rngs.params(), (dout,)))
        self.batchnorm = bridge.LinenToNNX(nn.BatchNorm(use_running_average=True), rngs=rngs)
        self.rngs = rngs

      def __call__(self, x):
        x = self.nn_dense1(x) + self.b
        return self.batchnorm(x)

    x = jax.random.normal(jax.random.key(0), (2, 4))
    model = NNXOuter(3, rngs=nnx.Rngs(0))
    gdef_before_shaped_init, _ = nnx.split(model)
    gdef_full, state = bridge.shaped_init(model, x)
    assert gdef_before_shaped_init != gdef_full
    y = model(x)
    k, b = state.nn_dense1.states.params.kernel.value, state.b.value
    np.testing.assert_allclose(y, x @ k + b, rtol=1e-5)
    assert gdef_full == nnx.graphdef(model)  # static data is stable now

if __name__ == '__main__':
  absltest.main()
