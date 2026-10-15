import time


def log_eval(
    eval_idx: int,
    num_evals: int,
    environment_steps: int,
    training_steps: int,
    red_mean_reward: float,
    blue_mean_reward: float,
    mean_episode_length: float,
    mean_episode_duration: float,
    eval_outcomes: dict[str, int],
    train_start_time: float,
    line_width: int = 80,
):
    """Print progress after one training/evaluation cycle."""

    elapsed_seconds = int(time.time() - train_start_time)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_seconds, 60)
    elapsed_hours, elapsed_minutes = divmod(elapsed_minutes, 60)
    elapsed_str = f"{elapsed_hours:02d}:{elapsed_minutes:02d}:{elapsed_seconds:02d}"

    header = f"Eval {eval_idx}/{num_evals}"
    print("")
    print("#" * line_width)
    print(f"\033[1m{header.center(line_width)}\033[0m")
    print(f"{f'Environment steps: {environment_steps}/{training_steps}'.center(line_width)}")
    print(f"{f'Red mean reward: {red_mean_reward:.2f}'.center(line_width)}")
    print(f"{f'Blue mean reward: {blue_mean_reward:.2f}'.center(line_width)}")
    episode_str = f"Mean episode length: {mean_episode_length:.1f} steps ({mean_episode_duration:.2f}s simulated)"
    print(episode_str.center(line_width))
    counts = {name: int(count) for name, count in eval_outcomes.items()}
    # Contact overlaps the win/timeout categories; it is not an extra outcome.
    total = sum(counts[name] for name in ("red", "blue", "both_failed", "timeout"))
    for name, label in (("red", "Red wins"), ("blue", "Blue wins"),
                        ("both_failed", "Both failed"), ("timeout", "Timeouts")):
        outcome_str = f"{label}: {counts[name]}/{total} ({100 * counts[name] / total:.1f}%)"
        print(outcome_str.center(line_width))
    contact_str = f"Contact: {counts['contact']}/{total} ({100 * counts['contact'] / total:.1f}%)"
    print(contact_str.center(line_width))
    print(f"{f'Time elapsed: {elapsed_str}'.center(line_width)}")
    print("-" * line_width)
