"""Test if creating training envs crashes."""
import sys
sys.path.insert(0, '/mnt/c/Users/sungj/Desktop/Sample/ygo-agent/vendor/ygo-agent')

import ygoenv
from ygoai.utils import init_ygopro

deck_dir = '/tmp'
code_list = '/mnt/c/Users/sungj/Desktop/Sample/ygo-agent/data/code_list.txt'

deck, deck_names = init_ygopro('YGOPro-v1', 'english', deck_dir, code_list, return_deck_names=True)
print(f"deck ok, names: {deck_names[:3]}...", flush=True)

for n_envs, n_threads in [(2,1), (8,4), (16,8), (32,16)]:
    print(f"Creating {n_envs} envs with {n_threads} threads...", flush=True)
    envs = ygoenv.make(
        task_id='YGOPro-v1',
        env_type='gymnasium',
        num_envs=n_envs,
        num_threads=n_threads,
        thread_affinity_offset=-1,
        seed=0,
        deck1=deck,
        deck2=deck,
        max_options=24,
        n_history_actions=32,
        async_reset=False,
        greedy_reward=False,
        play_mode='self',
        timeout=600,
        oppo_info=False,
    )
    obs, info = envs.reset()
    print(f"  envs.reset() ok, obs shape: {obs['cards_'].shape}", flush=True)
    envs.close()
    print(f"  closed ok", flush=True)
