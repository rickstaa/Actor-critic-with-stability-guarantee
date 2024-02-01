"""OscillatorComp Environment Translation Validation Script

This script assesses the translation fidelity of the 'OscillatorComp' environment from
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
from envs.oscillator_complicated import oscillator as OscillatorComp  # noqa: E402
from prettytable import PrettyTable  # noqa: E402
import textwrap  # noqa: E402

STEPS = 10
SEED = 0
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    print("=== OscillatorComp Environment Translation Validation ===")
    print(
        textwrap.dedent(
            f"""
            Welcome to the OscillatorComp environment translation validation script. This
            script initializes the OscillatorComp environment and performs '{STEPS}' steps
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

    # Initialize OscillatorComp environment.
    # NOTE: The state is set directly due to non-deterministic behaviour across different
    # numpy and gym versions.
    env_cost = OscillatorComp()
    env_cost.seed(SEED)
    env_cost = env_cost.unwrapped
    env_cost.reset()
    env_cost.unwrapped.state = np.array(
        [
            1.0046369,
            2.1169252,
            0.01870239,
            0.01570958,
            2.0646853,
            1.2112826,
            1.9610515,
            0.34511724,
        ],
        dtype=np.float32,
    )

    # Create a pretty table to display the results in.
    table = PrettyTable()
    table.field_names = [
        "Step",
        "Obs1",
        "Obs2",
        "Obs3",
        "Obs4",
        "Reward",
        "Done",
        "Reference",
        "State of Interest",
    ]

    # Perform N steps for the stable-gym environment comparison.
    # NOTE: Use the same action as in the stable-gym package.
    df = pd.DataFrame(
        columns=[
            "Step",
            "Obs1",
            "Obs2",
            "Obs3",
            "Obs4",
            "Reward",
            "Done",
            "Reference",
            "StateOfInterest",
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

        # Store the results in a table and dataframe.
        reference = info["reference"]
        state_of_interest = info["state_of_interest"]
        table.add_row(
            [
                i,
                observation[0],
                observation[1],
                observation[2],
                observation[3],
                reward,
                done,
                reference,
                state_of_interest,
            ]
        )
        df = df.append(
            {
                "Step": i,
                "Obs1": observation[0],
                "Obs2": observation[1],
                "Obs3": observation[2],
                "Obs4": observation[3],
                "Reward": reward,
                "Done": done,
                "Reference": reference,
                "StateOfInterest": state_of_interest,
            },
            ignore_index=True,
        )

    # Save the results to a CSV file.
    csv_file_path = os.path.join(
        SCRIPT_DIR, "results/oscillatorComplicated_translation_validation.csv"
    )
    df.to_csv(csv_file_path, index=False)

    # Print the results.
    print(table)

    env_cost.close()
    print(f"\nValidation results table created and saved to '{csv_file_path}'.")
