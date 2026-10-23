"""Atari environment construction, shared by training, evaluation and testing.

Frame-skip, exactly once
------------------------
``ALE/*-v5`` already applies ``frameskip=4`` internally, and
``AtariPreprocessing`` applies its own ``frame_skip``.  Using both defaults
would skip 16 frames per agent step.  We therefore build the raw env with
``frameskip=1`` and let ``AtariPreprocessing`` do the skipping (frame_skip=4),
because the wrapper additionally max-pools the last two raw frames, which
removes the Atari sprite flicker that plain ALE frame-skipping leaves in.

Sticky actions (``repeat_action_probability=0.25``) are the ``-v5`` default and
are kept: they are part of the benchmark this lab is graded on.  They are
driven by the ALE's own RNG, which is seeded from ``env.reset(seed=...)``, so a
fixed seed still reproduces a run exactly.
"""

from functools import partial

import gymnasium as gym
import numpy as np

try:  # ale_py >= 0.10 no longer auto-registers on import
    import ale_py
    gym.register_envs(ale_py)
except (ImportError, AttributeError):  # pragma: no cover - older ale-py
    pass

# gymnasium >= 1.0 renamed FrameStack -> FrameStackObservation
try:
    from gymnasium.wrappers import FrameStackObservation as _FrameStack
    _FRAMESTACK_KW = {}
except ImportError:  # pragma: no cover - gymnasium 0.29
    from gymnasium.wrappers import FrameStack as _FrameStack
    _FRAMESTACK_KW = {}

from gymnasium.wrappers import AtariPreprocessing


FRAME_SIZE = 84
FRAME_STACK = 4


def make_atari_env(env_id, seed=None, render_mode=None, frame_stack=FRAME_STACK,
                   noop_max=30, max_episode_frames=None):
    """A single preprocessed Atari env returning uint8 (frame_stack, 84, 84).

    Rewards are **not** clipped here -- the training loop clips its own copy and
    keeps the raw game score for logging, which is what the report needs.
    """
    kwargs = dict(frameskip=1)  # see module docstring
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if max_episode_frames is not None:
        kwargs["max_num_frames_per_episode"] = max_episode_frames

    env = gym.make(env_id, **kwargs)
    env = AtariPreprocessing(
        env,
        noop_max=noop_max,
        frame_skip=4,
        screen_size=FRAME_SIZE,
        terminal_on_life_loss=False,   # handled in the TD target, not by the env
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,               # keep uint8
    )
    env = _FrameStack(env, frame_stack, **_FRAMESTACK_KW)
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    return env


def make_vector_env(env_id, num_envs, frame_stack=FRAME_STACK, asynchronous=True):
    """`num_envs` preprocessed Atari envs for parallelized rollout.

    Autoreset is pinned to ``NEXT_STEP`` (the gymnasium >= 1.0 default): the
    step that ends an episode returns the *real* final observation, and the
    following step returns the reset observation while ignoring the action.
    The training loop drops that filler step instead of storing it.  The
    alternative (``SAME_STEP``) hides the final observation in
    ``info["final_observation"]``, which is easy to get wrong.

    ``functools.partial`` (not a closure) is used so the env factories stay
    picklable -- Windows spawns subprocesses rather than forking them.
    """
    fns = [partial(make_atari_env, env_id, None, None, frame_stack) for _ in range(num_envs)]
    if not asynchronous:
        return gym.vector.SyncVectorEnv(fns)
    if hasattr(gym.vector, "AutoresetMode"):
        return gym.vector.AsyncVectorEnv(fns, autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP)
    return gym.vector.AsyncVectorEnv(fns)  # pragma: no cover - gymnasium 0.29


def get_lives(info, index=None):
    """Read the ALE life counter out of a (possibly vectorized) info dict."""
    lives = info.get("lives", None)
    if lives is None:
        return None
    if index is None:
        return int(lives)
    return int(np.asarray(lives).reshape(-1)[index])
