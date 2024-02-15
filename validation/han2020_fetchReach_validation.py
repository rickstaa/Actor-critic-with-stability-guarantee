"""FetchReach Environment Translation Validation Script

This script assesses the translation fidelity of the 'FetchReach' environment from
the 'Actor-critic-with-stability-guarantee' repository to the 'stable_gym' package. It
executes a predefined number of steps within the environment and records the outcomes in
a CSV file for comparison.

Refer to the README.md in this directory for detailed usage instructions.
"""
import numpy as np
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import gym
from prettytable import PrettyTable  # noqa: E402
import textwrap  # noqa: E402
import mujoco_py

STEPS = 10
SEED = 0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    print("=== FetchReach Environment Translation Validation ===")
    print(
        textwrap.dedent(
            f"""
            Welcome to the FetchReach environment translation validation script. This
            script initializes the FetchReach environment and performs '{STEPS}' steps
            to validate the translations in the stable-gym package.

            Please compare the output of this script with the equivalent script in the
            'Actor-critic-with-stability-guarantee' repository. Matching outputs
            indicate correct translations.

            Note: These script only compares step outputs, not reset outputs, due to
            non-deterministic behaviour between different numpy and gym versions used in
            both packages.
            """
        )
    )

    # Initialize FetchReach environment.
    # NOTE: The state is set directly due to non-deterministic behaviour across different
    # numpy and gym versions.
    env_cost = gym.make("FetchReach-v1", reward_type="dense")
    env_cost.seed(SEED)
    env_cost.unwrapped.initial_state = mujoco_py.MjSimState(
            time=0.4000000000000003,
            qpos=np.array(
                [
                    4.04899887e-01,
                    4.80000000e-01,
                    2.79906896e-07,
                    -2.10804408e-05,
                    1.80448057e-10,
                    6.00288106e-02,
                    9.67580396e-03,
                    -8.28231087e-01,
                    -3.05625957e-03,
                    1.44397975e00,
                    2.53423937e-03,
                    9.55099996e-01,
                    5.96093593e-03,
                    1.97805133e-04,
                    7.15193042e-05,
                ]
            ),
            qvel=np.array(
                [
                    -8.20972730e-10,
                    -5.42827776e-13,
                    3.01801009e-07,
                    -2.06118511e-05,
                    1.60548710e-11,
                    7.22090854e-05,
                    7.12945378e-04,
                    9.39598373e-04,
                    -1.38720810e-03,
                    -1.63417143e-03,
                    1.15341321e-03,
                    1.24855991e-03,
                    -9.73707041e-04,
                    1.18331413e-04,
                    -5.71138070e-05,
                ]
            ),
            act=None,
            udd_state={},
        )
    env_cost.reset()
    env_cost.unwrapped.goal = np.array([1.37384575, 0.81794948, 0.54779444])

    # Create a pretty table to display the results in.
    obs_cols = [
        f"Obs{i+1}"
        for i in range(env_cost.observation_space.spaces["observation"].shape[0])
    ]
    achieved_goal_cols = [f"AchievedGoal{dim}" for dim in ['x', 'y', 'z']]
    desired_goal_cols = [f"DesiredGoal{dim}" for dim in ['x', 'y', 'z']]
    table = PrettyTable()
    table.field_names = [
        "Step",
        *obs_cols,
        "Reward",
        "Done",
        *achieved_goal_cols,
        *desired_goal_cols,
        "IsSuccess",
    ]

    # Perform N steps for the stable-gym environment comparison.
    # NOTE: Use the same action as in the stable-gym package.
    df = pd.DataFrame(
        columns=[
            "Step",
            *obs_cols,
            "Reward",
            "Done",
            *achieved_goal_cols,
            *desired_goal_cols,
            "IsSuccess",
        ]
    )
    for i in range(STEPS):
        delta = (
            (env_cost.action_space.high - env_cost.action_space.low)[0] / STEPS
        ) * i
        action = np.array(
            [
                env_cost.action_space.low[0] + delta,
                env_cost.action_space.high[1] - delta,
                env_cost.action_space.low[2] + delta,
                env_cost.action_space.high[3] - delta,
            ],
            dtype=np.float32,
        )
        observation, reward, done, info = env_cost.step(action)
        reward = np.abs(reward)  # Convert negative rewards to positive for comparison.

        # Store the results in a table and dataframe.
        table.add_row(
            [
                i,
                *observation["observation"],
                reward,
                done,
                *observation["achieved_goal"],
                *observation["desired_goal"],
                info["is_success"],
            ]
        )
        obs_dict = {
            f"Obs{i+1}": obs for i, obs in enumerate(observation["observation"])
        }
        achieved_goal_dict = {
            f"AchievedGoal{dim}": obs
            for dim, obs in zip(["x", "y", "z"], observation["achieved_goal"])
        }
        desired_goal_dict = {
            f"DesiredGoal{dim}": obs
            for dim, obs in zip(["x", "y", "z"], observation["desired_goal"])
        }
        data = {
            "Step": np.int64(i),
            "Reward": reward,
            "Done": done,
            **obs_dict,
            **achieved_goal_dict,
            **desired_goal_dict,
            "IsSuccess": info["is_success"],
        }
        df = df.append(data, ignore_index=True)

    # Save the results to a CSV file.
    csv_file_path = os.path.join(
        SCRIPT_DIR, "results/fetchReach_translation_validation.csv"
    )
    df.to_csv(csv_file_path, index=False)

    # Print the results.
    print(table)

    env_cost.close()
    print(f"\nValidation results table created and saved to '{csv_file_path}'.")
