import sys
sys.path.insert(0, '/mnt/c/Users/sungj/Desktop/Sample/ygo-agent/vendor/ygo-agent')
import jax, jax.numpy as jnp, flax
from dataclasses import asdict
from ygoai.rl.jax.agent import RNNAgent, ModelArgs

args_m1 = ModelArgs(num_layers=2, num_channels=128, use_history=True, card_mask=False,
    noam=True, action_feats=True, version=2, rnn_channels=512, rnn_type='lstm',
    film=True, oppo_info=False, rnn_shortcut=False, batch_norm=False,
    critic_width=128, critic_depth=3)

agent = RNNAgent(**asdict(args_m1), embedding_shape=None,
    dtype=jnp.float32, param_dtype=jnp.float32, switch=False, freeze_id=False)
print("agent created", flush=True)

rstate = agent.init_rnn_state(1)
sample_obs = {
    'cards_': jnp.zeros((1, 160, 41), dtype=jnp.uint8),
    'global_': jnp.zeros((1, 23), dtype=jnp.uint8),
    'actions_': jnp.zeros((1, 24, 12), dtype=jnp.uint8),
    'h_actions_': jnp.zeros((1, 32, 14), dtype=jnp.uint8),
    'mask_': jnp.zeros((1, 160, 14), dtype=jnp.uint8),
}
key = jax.random.PRNGKey(42)
variables = agent.init(key, sample_obs, rstate)
variables = flax.core.unfreeze(variables)
print("agent.init ok", flush=True)

import optax
tx = optax.MultiSteps(
    optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.inject_hyperparams(optax.adam)(learning_rate=0.0003, eps=1e-5),
    ), every_k_schedule=1,
)
tx = optax.apply_if_finite(tx, max_consecutive_errors=10)
print("tx ok", flush=True)

from ygoai.rl.jax.utils import TrainState
if 'batch_stats' not in variables:
    variables['batch_stats'] = {}
agent_state = TrainState.create(
    apply_fn=agent.apply, params=variables['params'],
    tx=tx, batch_stats=variables['batch_stats'])
print("TrainState ok", flush=True)

learner_devices = jax.devices()
agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
print("replicate ok — all init done", flush=True)
