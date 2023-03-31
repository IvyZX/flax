---
jupyter:
  jupytext:
    formats: ipynb,md
    main_language: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
---

<!-- #region id="6e9134fa" -->
# Save and load checkpoints

In this guide, you will learn about saving and loading your Flax checkpoints with [Orbax](https://github.com/google/orbax).

Orbax provides a variety of features related to saving and loading model data, which this guide will showcase:

*  Support to various array types and storage formats;
*  Asynchronous saving to reduce training wait time;
*  Versioning and automatic bookkeeping of past checkpoints;
*  Flexible [`transformations`](https://github.com/google/orbax/blob/main/docs/checkpoint.md#transformations) to tweak and load old checkpoints;
*  [`jax.sharding`](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)-based API to save and load in multi-host scenarios.

For backward-compatibility purposes, this guide will also show the equivalent calls in Flax legacy checkpointing APIs `flax.training.checkpoints`.

See [Orbax Checkpoint Documentation](https://github.com/google/orbax/blob/main/docs/checkpoint.md) for more details.

---
**_Ongoing Migration:_** 

After July 30 2023, Flax's legacy checkpointing APIs `flax.training.checkpoints` will enter maintainence mode in favor of [Orbax](https://github.com/google/orbax).

*  **If you are a new Flax user**: Please start with the new `orbax.checkpoint` API, as demonstrated in this guide. 

*  **If you have used legacy Flax APIs `flax.training.checkpoints`**: You have two options to consider:

   * **Recommended**: Migrate your API calls to `orbax.checkpoint` API following this simple [migration guide](https://flax.readthedocs.io/en/latest/guides/orbax_upgrade_guide.html).

   * **Transitory**: Try adding `flax.config.update('flax_use_orbax_checkpointing', True)` to your project, which will let your `flax.training.checkpoints` calls automatically use the Orbax backend to save your checkpoints. 
   
      * **Special conditions may apply - check out the [Orbax-as-backend troubleshoot section](#orbax-as-backend-troubleshoot) for any issues.** 
---

<!-- #endregion -->

<!-- #region id="5a2f6aae" -->
## Setup

Install/upgrade Flax and [Orbax](https://github.com/google/orbax). For JAX installation with GPU/TPU support, visit [this section on GitHub](https://github.com/google/jax#installation).
<!-- #endregion -->

```python tags=["skip-execution"]
# replace with `pip install flax` after release 0.6.9.
! pip install -U -qq "git+https://github.com/google/flax.git@main#egg=flax"
```

<!-- #region id="-icO30rwmKYj" -->
Note: Before running `import jax`, create eight fake devices to mimic [multi-host environment](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html?#aside-hosts-and-devices-in-jax) this notebook. Note that the order of imports is important here. The `os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'` command works only with the CPU backend. This means it won't work with GPU/TPU acceleration on if you're running this notebook in Google Colab. If you are already running the code on multiple devices (for example, in a 4x2 TPU environment), you can skip running the next cell.
<!-- #endregion -->

```python id="ArKLnsyGRxGv"
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
```

```python id="SJT9DTxTytjn"
from typing import Optional, Any
import shutil

import numpy as np
import jax
from jax import random, numpy as jnp

import flax
from flax import linen as nn
from flax.training import checkpoints, train_state
from flax import struct, serialization
import orbax.checkpoint

import optax
```

```python
ckpt_dir = 'tmp'

if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)  # Remove any existing checkpoints from the last notebook run.
```

<!-- #region id="40d434cd" -->
## Save checkpoints

In Orbax and Flax, you can save and load any given JAX [pytree](https://jax.readthedocs.io/en/latest/pytrees.html). This includes not only typical Python and NumPy containers, but also customized classes extended from [`flax.struct.dataclass`](https://flax.readthedocs.io/en/latest/api_reference/flax.struct.html#flax.struct.dataclass). That means you can store almost any data generated — not only your model parameters, but any arrays/dictionaries, metadata/configs, and so on.

Create a pytree with many data structures and containers, and play with it:
<!-- #endregion -->

```python id="56dec3f6" outputId="f1856d96-1961-48ed-bb7c-cb63fbaa7567"
# A simple model with one linear layer.
key1, key2 = random.split(random.PRNGKey(0))
x1 = random.normal(key1, (5,))      # A simple JAX array.
model = nn.Dense(features=3)
variables = model.init(key2, x1)

# Flax's TrainState is a pytree dataclass and is supported in checkpointing.
# Define your class with `@flax.struct.dataclass` decorator to make it compatible.
tx = optax.sgd(learning_rate=0.001)      # An Optax SGD optimizer.
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    tx=tx)
# Perform a simple gradient update similar to the one during a normal training workflow.
state = state.apply_gradients(grads=jax.tree_map(jnp.ones_like, state.params))

# Some arbitrary nested pytree with a dictionary, a string, and a NumPy array.
config = {'dimensions': np.array([5, 3]), 'name': 'dense'}

# Bundle everything together.
ckpt = {'model': state, 'config': config, 'data': [x1]}
ckpt
```

<!-- #region id="6fc59dfa" -->
Now save the checkpoint with an Orbax `PyTreeCheckpointer`, directly to `tmp/orbax/single_save` directory.

Note that an optional `save_args` is provided. This is recommended for performance speedups, as it bundles smaller arrays in your pytree to a single large file instead of multiple smaller files.
<!-- #endregion -->

```python id="0pp4QtEqW9k7"
from flax.training import orbax_utils

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save('tmp/orbax/single_save', ckpt, save_args=save_args)
```

To use versioning and automatic bookkeeping features, you need to wrap an Orbax `CheckpointManager` over the `PyTreeCheckpointer`. 

Also provide an `CheckpointManagerOptions` that customizes your needs, such as how often and on what criteria you prefer old checkpoints be deleted. See [documentation](https://github.com/google/orbax/blob/main/docs/checkpoint.md#checkpointmanager) for a full list of options offered.

`CheckpointManager` should be placed at top-level outside your training steps, to overall manage your saves.

```python id="T6T8V4UBXB1R" outputId="b7132933-566d-440d-c34e-c5468d87cbdc"
options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    'tmp/orbax/managed', orbax_checkpointer, options)

# Inside a training loop
for step in range(5):
    # ... do your training ...
    checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

os.listdir('tmp/orbax/managed')  # since max_to_keep=2, only step 3 and 4 are retained
```

<!-- #region id="OQkUOkHVW_4e" -->
Alternatively you could save with the legacy Flax checkpointing utilities. Note that this provides less management features than Orbax's `CheckpointManagerOptions`.
<!-- #endregion -->

```python id="4cdb35ef" outputId="6d849273-15ce-4480-8864-726d1838ac1f"
# Import Flax Checkpoints.
from flax.training import checkpoints

checkpoints.save_checkpoint(ckpt_dir='tmp/flax-checkpointing',
                            target=ckpt,
                            step=0,
                            overwrite=True,
                            keep=2)
```

<!-- #region id="6b658bd1" -->
## Restore checkpoints

In Orbax, call `.restore()` for either `PyTreeCheckpointer` or `CheckpointManager` will restore your checkpoint, in the raw pytree format.
<!-- #endregion -->

```python id="WgRJj3wjXIaN" outputId="b4af1ef4-f22f-459b-bdca-2e6bfa16c08b"
raw_restored = orbax_checkpointer.restore('tmp/orbax/single_save')
raw_restored
```

Note that the `step` number is required for `CheckpointManger`. You could also use `.latest_step()` to find the latest step available.

```python
step = checkpoint_manager.latest_step()  # step = 4
checkpoint_manager.restore(step)
```

<!-- #region id="VKJrfSyLXGrc" -->
Alternatively you could load the checkpoint with the legacy Flax checkpointing utilities.

Note that With the migration to Orbax in progress, `restore_checkpoint` can automatically identify whether a checkpoint is saved in the legacy Flax format or with an Orbax backend, and restore the pytree correctly. Therefore, adding `flax.config.update('flax_use_orbax_checkpointing', True)` won't hurt your ability to restore old checkpoints.
<!-- #endregion -->

```python id="150b20a0" outputId="85ffceca-f38d-46b8-e567-d9d38b7885f9"
raw_restored = checkpoints.restore_checkpoint(ckpt_dir='tmp/flax-checkpointing', target=None)
raw_restored
```

<!-- #region id="987b981f" -->
## Restore with custom dataclasses

Note that all the pytrees restored above are raw dictionaries, whereas the original pytrees contain custom dataclasses like `TrainState` and `optax` states. That is because when restoring the pytree, the program does not yet know which structure it once belongs to.

To resolve this, you should provide an example pytree to let the Orbax or Flax know exactly what structure it should restore to. This example should introduce any custom Flax dataclasses explicitly, and have the same structure as the saved checkpoint.

> Side note: Data that was a JAX NumPy array (`jnp.array`) will be restored as a NumPy array (`numpy.array`). This would not affect your work because JAX will [automatically convert](https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html) NumPy arrays to JAX arrays once the computation starts.
<!-- #endregion -->

```python id="58f42513" outputId="110c6b6e-fe42-4179-e5d8-6b92d355e11b"
empty_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=jax.tree_map(np.zeros_like, variables['params']),  # values of the tree leaf doesn't matter
    tx=tx,
)
empty_config = {'dimensions': np.array([0, 0]), 'name': ''}
target = {'model': empty_state, 'config': empty_config, 'data': [jnp.zeros_like(x1)]}
state_restored = orbax_checkpointer.restore('tmp/orbax/single_save', item=target)
state_restored
```

Alternatively, restore from Orbax `CheckpointManager` and from legacy Flax code:

```python
checkpoint_manager.restore(4, items=target)
```

```python
checkpoints.restore_checkpoint(ckpt_dir='tmp/flax-checkpointing', target=target)
```

It's often recommended to refactor out the process of initializing a checkpoint's structure (for example, a [`TrainState`](https://flax.readthedocs.io/en/latest/flip/1009-optimizer-api.html?#train-state)), so that saving/loading is easier and less error-prone. This is because functions and complex objects like `apply_fn` and `tx` (optimizer) cannot be serialized into the checkpoint file and must be initiated by code.

<!-- #region id="136a300a" -->
### Restore when checkpoint structures differ

During your development, at times you change your model, or you add and remove fields to tweak it better for your needs. That means your checkpoint structure will change! How can your old data be loaded to your new code?

Below showcases a simple example - a `TrainStatePlus` extended from `TrainState` that contains an extra field `batch_stats`. In real practice, you may need this when doing [batch normalization](https://flax.readthedocs.io/en/latest/guides/batch_norm.html).

Let's store the new `TrainStatePlus` as step 5. Remember that step 4 has the old `TrainState`.
<!-- #endregion -->

```python id="be65d4af" outputId="4fe776f0-65f8-4fc4-d64a-990520b36dce"
class CustomTrainState(train_state.TrainState):
    batch_stats: Any = None

custom_state = CustomTrainState.create(
    apply_fn=state.apply_fn,
    params=state.params,
    tx=state.tx,
    batch_stats=np.arange(10),
)

custom_ckpt = {'model': custom_state, 'config': config, 'data': [x1]}
# Use a custom state to read the old `TrainState` checkpoint.
custom_target = {'model': custom_state, 'config': None, 'data': [jnp.zeros_like(x1)]}


# Save it in Orbax.
custom_save_args = orbax_utils.save_args_from_target(custom_ckpt)
checkpoint_manager.save(5, custom_ckpt, save_kwargs={'save_args': custom_save_args})
```

<!-- #region id="379c2255" -->
It is recommended to keep your checkpoints up to date with your pytree dataclass definitions, but things happens and you might be forced to restore checkpoints with incompatible reference objects at runtime. 

When this happens, the checkpoint restoration will try to respect the structure of the reference when given. Below showcases a few common scenarios.
<!-- #endregion -->

#### When reference object is partial

If your reference object is a subtree of your checkpoint, the restoration will ignore the additional field(s) and restore a checkpoint with same structure as the reference. 

Like in the example below, the `batch_stats` field in `CustomTrainState` was ignored, and the checkpoint was restored as a `TrainState`.

This can also be useful for reading only part of your checkpoint.

```python
restored = checkpoint_manager.restore(5, items=target)
assert not hasattr(restored, 'batch_stats')
assert type(restored['model']) == train_state.TrainState
restored
```

#### When checkpoint is partial

On the other hand, if the reference object contains a value that is not available in the checkpoint, the checkpointing code will by default warn that some data is not compatible.

To bypass the error, you need to pass an Orbax [`transform`](https://github.com/google/orbax/blob/main/docs/checkpoint.md#transformations) that teaches Orbax how to conform this checkpoint into the structure of `custom_target`.

In this case, we pass a default `{}` that lets Orbax use values in `custom_target` to fill in the blank. This lets us restore an old checkpoint into a new data structure, the `CustomTrainState`.

```python
try:
    checkpoint_manager.restore(4, items=custom_target)
except KeyError as e:
    print(f'KeyError when target state has an unmentioned field: {e}')
    print('')

# Step 4 is an original TrainState, without "batch_stats"
restored = checkpoint_manager.restore(4, items=custom_target, 
                                      restore_kwargs={'transforms': {}})
assert type(restored['model']) == CustomTrainState
np.testing.assert_equal(restored['model'].batch_stats, 
                        custom_target['model'].batch_stats)
restored
```

If you already saved your checkpoints with an Orbax backend, you can use `orbax_transforms` to access this `transforms` arg in the Flax API.

```python id="29fd1e33" outputId="cdbb9247-d1eb-4458-aa83-8db0332af7cb"
# Save in Flax-with-Orbax-backend.
flax.config.update('flax_use_orbax_checkpointing', True)
checkpoints.save_checkpoint(ckpt_dir='tmp/flax-checkpointing',
                            target=ckpt,
                            step=4,
                            overwrite=True,
                            keep=2)

checkpoints.restore_checkpoint('tmp/flax-checkpointing', target=custom_target, step=4,
                               orbax_transforms={})
```

In legacy Flax checkpoint APIs, similar things are doable too, but not as flexible as [Orbax Transformations](https://github.com/google/orbax/blob/main/docs/checkpoint.md#transformations). You need to first restore the checkpoint to a raw dict with `target=None`, modify the structure accordingly, and then deserialize it back to original target.

```python
# Save in legacy Flax.
flax.config.update('flax_use_orbax_checkpointing', False)
checkpoints.save_checkpoint(ckpt_dir='tmp/flax-checkpointing',
                            target=ckpt,
                            step=5,
                            overwrite=True,
                            keep=2)

# Pass no target to get a raw state dictionary first.
raw_state_dict = checkpoints.restore_checkpoint('tmp/flax-checkpointing', target=None, step=5)
# Add/remove fields as needed.
raw_state_dict['model']['batch_stats'] = np.flip(np.arange(10))
# Restore the classes with correct target now
flax.serialization.from_state_dict(custom_target, raw_state_dict)
```

<!-- #region id="a6b39501" -->
## Asynchronized checkpointing

Checkpointing is I/O heavy, and if you have a large amount of data to save, it may be worthwhile to put it into a background thread, while continuing with your training.

You can do this by creating an [`orbax.checkpoint.AsyncCheckpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/async_checkpointer.py) in place of the `orbax.checkpoint.PyTreeCheckpointer`.

Note: You should use the same `async_checkpointer` to handle all your async saves across your training steps, so that it can make sure that a previous async save is done before the next one begins. This enables bookkeeping, such as `keep` (the number of checkpoints) and `overwrite` to be consistent across steps.

Whenever you want to explicitly wait until an async save is done, you can call `async_checkpointer.wait_until_finished()`.
<!-- #endregion -->

```python id="85be68a6" outputId="aefce94c-8bae-4355-c142-05f2b61c39e2"
# `orbax.checkpoint.AsyncCheckpointer` needs some multi-process initialization, because it was
# originally designed for multi-process large model checkpointing.
# For Python notebooks or other single-process setting, just set up with `num_processes=1`.
# Refer to https://jax.readthedocs.io/en/latest/multi_process.html#initializing-the-cluster
# for how to set it up in multi-process scenarios.
jax.distributed.initialize("localhost:8889", num_processes=1, process_id=0)

async_checkpointer = orbax.checkpoint.AsyncCheckpointer(
    orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=50)

# Save your job:
async_checkpointer.save('tmp/orbax/single_save_async', ckpt, save_args=save_args)
# ... Continue with your work...

# ... Until a time when you want to wait until the save completes:
async_checkpointer.wait_until_finished()  # Blocks until the checkpoint saving is completed.
async_checkpointer.restore('tmp/orbax/single_save_async', item=target)
```

<!-- #region id="QpuTCeMVXOBn" -->
If you are using Orbax `CheckpointManager`, just pass in the async_checkpointer when initializing it. Then, in practice, call `async_checkpoint_manager.wait_until_finished()` instead.
<!-- #endregion -->

```python
async_checkpoint_manager = orbax.checkpoint.CheckpointManager(
    'tmp/orbax/managed_async', async_checkpointer, options)
async_checkpoint_manager.wait_until_finished()
```

<!-- #region id="13e93db6" -->
## Multi-host/multi-process checkpointing

JAX provides a few ways to scale up your code on multiple hosts at the same time. This usually happens when the number of devices (CPU/GPU/TPU) is so large that different devices are managed by different hosts (CPU). To get started on JAX in multi-process settings, check out [Using JAX in multi-host and multi-process environments](https://jax.readthedocs.io/en/latest/multi_process.html) and the [distributed array guide](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html).

In the [Single Program Multi Data (SPMD)](https://jax.readthedocs.io/en/latest/glossary.html#term-SPMD) paradigm with JAX [`pjit`](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html), a large multi-process array can have its data sharded across different devices (check out the `pjit` [JAX-101 tutorial](https://jax.readthedocs.io/en/latest/jax-101/08-pjit.html)). When a multi-process array is serialized, each host dumps its data shards to a single shared storage, such as a Google Cloud bucket.

Orbax supports saving and loading pytrees with multi-process arrays in the same fashion as single-process pytrees. However, it's recommended to use the asynchronized [`orbax.AsyncCheckpointer`](https://github.com/google/orbax/blob/main/orbax/checkpoint/async_checkpointer.py) to save large multi-process arrays on another thread, so that you can perform computation alongside the saves. With pure Orbax, saving checkpoints in a multiprocess context uses the same API as in a single process context.
<!-- #endregion -->

```python id="ubdUvyMrhD-1"
from jax.sharding import PartitionSpec, NamedSharding

# Create an array sharded across multiple devices.
mesh_shape = (4, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = jax.sharding.Mesh(devices, ('x', 'y'))

mp_array = jax.device_put(np.arange(8 * 2).reshape(8, 2),
                          NamedSharding(mesh, PartitionSpec('x', 'y')))

# Make it a pytree, just because we can.
mp_ckpt = {'model': mp_array}
```

```python
async_checkpoint_manager.save(0, mp_ckpt)
async_checkpoint_manager.wait_until_finished()
```

When restoring a checkpoint with multi-process arrays, you need to specify what `sharding` each array should be restored back to. Otherwise, they will be restored as large `np.array`s on process 0, costing time and memory.

(In this notebook, since we are on single-process, it will be restored as `np.array` even if we provide shardings.)

Orbax allows you to specify this by passing a pytree of `sharding`s in `restore_args`. If you already have a reference pytree that has all the arrays with the right sharding, you can use `orbax_utils.restore_args_from_target` to transform it into the `restore_args` that Orbax needs.

```python
# The reference doesn't need to be as large as your checkpoint!
# Just make sure it has the `.sharding` you want.
mp_smaller = jax.device_put(np.arange(8).reshape(4, 2),
                            NamedSharding(mesh, PartitionSpec('x', 'y')))
ref_ckpt = {'model': mp_smaller}

restore_args = orbax_utils.restore_args_from_target(ref_ckpt)
async_checkpoint_manager.restore(
    0, items=ref_ckpt, restore_kwargs={'restore_args': restore_args})
```

<!-- #region id="edc355ce" -->
### Legacy Flax: use `save_checkpoint_multiprocess`

In legacy Flax, to save multi-process arrays, use [`flax.training.checkpoints.save_checkpoint_multiprocess()`](https://flax.readthedocs.io/en/latest/api_reference/flax.training.html#flax.training.checkpoints.save_checkpoint_multiprocess) in place of `save_checkpoint()` and with the same arguments.

If your checkpoint is too large, you can specify `timeout_secs` in the manager and give it more time to finish writing.
<!-- #endregion -->

```python id="5d10039b" outputId="901bb097-0899-479d-b9ae-61dae79e7057"
async_checkpointer = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=50)
checkpoints.save_checkpoint_multiprocess(ckpt_dir,
                                         mp_ckpt,
                                         step=3,
                                         overwrite=True,
                                         keep=4,
                                         orbax_checkpointer=async_checkpointer)
```

```python id="a9f9724c" outputId="393c4a0e-8a8c-4ca6-c609-93c8bab38e75"
mp_restored = checkpoints.restore_checkpoint(ckpt_dir,
                                             target=ref_ckpt,
                                             step=3,
                                             orbax_checkpointer=async_checkpointer)
mp_restored
```

## Orbax-As-Backend Troubleshoot

As an intermediate stage of the migration, `flax.training.checkpoints` APIs will start to use Orbax as their backend when saving checkpoints starting from May 15, 2023.

Checkpoints saved with Orbax backend can be readable by either `flax.training.checkpoints.restore_checkpoint` or a `orbax.checkpoint.PyTreeCheckpointer`.

Code-wise, this is equivalent to setting the config flag [`flax.config.flax_use_orbax_checkpointing`](https://github.com/google/flax/blob/main/flax/configurations.py#L103) default to `True`. You could overwrite this value in your project with `flax.config.update('flax_use_orbax_checkpointing', <BoolValue>)` at any time.

In general, this automatic migration will not affect most users. However, you may encounter issues if your API usage follows some specific pattern. Check out the sections below for troubleshooting.


### If your devices hangs when writing checkpoints

If you are running in multi-host environment (usually anything larger than 8 TPU devices) and your devices hang when writing checkpoints, please check out if your code is in the following pattern (aka. `save_checkpoint` only ran on host 0):

```
if jax.process_index() == 0:
  flax.training.checkpoints.save_checkpoint(...)
```

Unfortunately this is a legacy pattern that we plan to deprecate our support for, because in multi-process environment, the checkpointing code should coordinate among hosts instead of being triggered only on the host 0. Replacing the code above with the following should resolve the hang issue:

```
flax.training.checkpoints.save_checkpoint_multiprocess(...)
```


### If you don't save pytrees

Orbax uses `PyTreeCheckpointHandler` to save checkpoints, which means they only save pytrees. If you want to save singular arrays or numbers, you have two options:

1. Use `orbax.ArrayCheckpointHandler` to save them following [this migration section](https://flax.readthedocs.io/en/latest/guides/orbax_upgrade_guide.html#saving-loading-a-single-jax-or-numpy-array).

1. Wrap it inside a pytree and save as usual.
