import logging
from gym.envs.registration import register, registry

logger = logging.getLogger(__name__)

# Register panda_openai_sim environments
try:

    # Import panda_openai_sim environments
    from panda_openai_sim.envs.task_envs import (
        PandaSlideEnv,
        PandaPickAndPlaceEnv,
        PandaReachEnv,
        PandaPushEnv,
    )

    # Initialize ros time
    import rospy
    rospy.init_node("actor_critic_with_stability_guarantee")

    # Re-register panda_openai_sim environments such that the parameters can be changed
    # Main task env (Contains all input arguments)
    env_id = "PandaTask-v0"
    del registry.env_specs[env_id]
    register(
        id=env_id,
        entry_point="panda_openai_sim.envs.task_envs:PandaTaskEnv",
        # kwargs=kwargs,
        max_episode_steps=50,
    )

    # Slide
    env_id = "PandaSlide-v0"
    del registry.env_specs[env_id]
    register(
        id=env_id,
        entry_point="panda_openai_sim.envs.task_envs:PandaSlideEnv",
        # kwargs=kwargs,
        max_episode_steps=50,
    )

    # Pick and Place
    env_id = "PandaPickAndPlace-v0"
    del registry.env_specs[env_id]
    register(
        id=env_id,
        entry_point="panda_openai_sim.envs.task_envs:PandaPickAndPlaceEnv",
        # kwargs=kwargs,
        max_episode_steps=50,
    )

    # Reach
    env_id = "PandaReach-v0"
    del registry.env_specs[env_id]
    register(
        id=env_id,
        entry_point="panda_openai_sim.envs.task_envs:PandaReachEnv",
        # kwargs=kwargs,
        max_episode_steps=50,
    )

    # Push
    env_id = "PandaPush-v0"
    del registry.env_specs[env_id]
    register(
        id=env_id,
        entry_point="panda_openai_sim.envs.task_envs:PandaPushEnv",
        # kwargs=kwargs,
        max_episode_steps=50,
    )
except ImportError as e:

    # Check what went wrong and print log message
    if "dynamic module" in e.args[0].lower():
        print(
            "The 'panda_openai_sim' environments could not be registered. It appears "
            "that you did not build/source the required ROS for python 3. Please build "
            "and build these packages if you want to use the 'panda_openai_sim' "
            "environments."
        )
    elif "no module named 'rospy'" in e.args[0].lower():
        print(
            "The 'panda_openai_sim' environments could not be registered. It appears "
            "that you did not install/source the ROS. Please install/source ROS if you "
            "want to use the 'panda_openai_sim' environments."
        )
    else:
        print(
            "The 'panda_openai_sim' environments could not be registered. Please make "
            "sure you sourced the catkin_ws if you want to use the 'panda_openai_sim' "
            "environments."
        )

# Register Atari environments
# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['pong']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}NoFrameskip-v5'.format(name),
            entry_point='ENV.env.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

# Register classic environments
register(
    id='CartPolecons-v0',
    entry_point='ENV.env.classic_control:CartPoleEnv_cons',
    max_episode_steps=2500,
)

register(
    id='CartPolecost-v0',
    entry_point='ENV.env.classic_control:CartPoleEnv_cost',
    max_episode_steps=2500,
)

register(
    id='Carcost-v0',
    entry_point='ENV.env.classic_control:CarEnv',
    max_episode_steps=600,
)

# Register mujoco environments
register(
    id='HalfCheetahcons-v0',
    entry_point='ENV.env.mujoco:HalfCheetahEnv_lya',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)
register(
    id='HalfCheetahcost-v0',
    entry_point='ENV.env.mujoco:HalfCheetahEnv_cost',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Pointcircle-v0',
    entry_point='ENV.env.mujoco:PointEnv',
    max_episode_steps=65,
)

register(
    id='Antcons-v0',
    entry_point='ENV.env.mujoco:AntEnv_cpo',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Antcost-v0',
    entry_point='ENV.env.mujoco:AntEnv_cost',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='Quadrotorcons-v0',
    entry_point='ENV.env.mujoco:QuadrotorEnv',
    max_episode_steps=512,
)
