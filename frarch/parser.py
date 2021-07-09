import argparse


def parse_arguments():
    """Parse command-line arguments to the experiment.
    Returns
    -------
    param_file : str
        The location of the parameters file.
    parameters : argparse.NameSpace
        options
    """
    parser = argparse.ArgumentParser(
        description="Run a SpeechBrain experiment",
    )
    parser.add_argument(
        "param_file",
        type=str,
        help="A yaml-formatted file using the extended YAML syntax. "
        "defined by SpeechBrain.",
    )

    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Run the experiment with only a few batches for all "
        "datasets, to ensure code runs without crashing.",
    )

    parser.add_argument(
        "--debug_batches",
        type=int,
        default=2,
        help="Number of batches to run in debug mode.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="The device to run the experiment on (e.g. 'cuda:0')",
    )

    parser.add_argument(
        "--noprogressbar",
        default=False,
        action="store_true",
        help="This flag disables the data loop progressbars.",
    )

    parser.add_argument(
        "--ckpt_interval_minutes",
        type=float,
        help="Amount of time between saving intra-epoch checkpoints "
        "in minutes. If non-positive, intra-epoch checkpoints are not saved.",
    )

    args = parser.parse_args().__dict__
    param_file = args.pop('param_file')

    return param_file, args
