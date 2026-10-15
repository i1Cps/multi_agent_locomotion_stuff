"""Browse a run's evaluation checkpoints in the MuJoCo viewer"""

from queue import Empty, SimpleQueue
import time

from absl import app, flags
import glfw
import jax
import mujoco
import mujoco.viewer

from .registry import PLAY_ENV_REGISTRY

_RUN_DIR = flags.DEFINE_string("run_dir", None, "Run folder; defaults to the newest run with eval checkpoints")
_EVAL = flags.DEFINE_integer("eval", None, "Initial eval number; defaults to the last saved eval", lower_bound=1)
_SEED = flags.DEFINE_integer("seed", 0, "Initial spawn seed", lower_bound=0)


def run_viewer(play_env, seed: int = 0):
    model = play_env.mj_model
    data = mujoco.MjData(model)
    spawn_seed = seed
    paused = False
    status = ""
    pending_keys = SimpleQueue()

    def reset_spawn():
        _, key = play_env.reset(data, jax.random.PRNGKey(spawn_seed))
        mujoco.mj_forward(model, data)
        return key

    rng = reset_spawn()

    # Called on the viewer thread. Loading and resetting happen on the main
    # thread, between simulation steps, including while playback is paused
    def key_callback(keycode):
        if keycode in (
            glfw.KEY_LEFT, 
            glfw.KEY_RIGHT, 
            glfw.KEY_DOWN,
            glfw.KEY_UP, 
            glfw.KEY_SPACE
        ):
            pending_keys.put(keycode)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=key_callback,
        show_left_ui=False, show_right_ui=False,
    ) as viewer:
        with viewer.lock():
            viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = model.camera("default").id

        last_overlay = None

        def update_overlay(message=""):
            """ Helper function to refresh the mujoco viewer overlay"""
            nonlocal last_overlay

            checkpoint = play_env.checkpoint_path
            elapsed_seconds = play_env.episode_steps * play_env.ctrl_dt
            episode_seconds = play_env.episode_length * play_env.ctrl_dt
            eval_label = (
                f"{checkpoint.name} "
                f"({play_env.checkpoint_index + 1}/{len(play_env.checkpoint_paths)})"
            )

            rows = [
                ("Run", checkpoint.parent.name),
                ("Eval", eval_label),
                ("Spawn seed", str(spawn_seed)),
                ("Playback", "PAUSED" if paused else "RUNNING"),
                ("Episode", f"{elapsed_seconds:.1f} / {episode_seconds:.1f} s"),
            ]
            if message:
                rows.append(("Status", message))

            labels = "\n".join(label for label, _ in rows)
            values = "\n".join(value for _, value in rows)
            overlay = (labels, values)

            if overlay != last_overlay:
                viewer.set_texts([
                    (
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        labels,
                        values,
                    ),
                    (
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                        "Left / Right\nSpace\nDown\nUp",
                        "Previous / next eval\nPause / resume\nReplay this spawn\nNext spawn",
                    ),
                ])
                last_overlay = overlay

        update_overlay()
        viewer.sync()

        while viewer.is_running():
            frame_start = time.perf_counter()

            # Handle queued keyboard input
            while True:
                try:
                    keycode = pending_keys.get_nowait()
                except Empty:
                    break
                if keycode == glfw.KEY_SPACE:
                    paused = not paused
                elif keycode in (glfw.KEY_LEFT, glfw.KEY_RIGHT):
                    direction = -1 if keycode == glfw.KEY_LEFT else 1
                    checkpoint_index = (play_env.checkpoint_index + direction) % len(play_env.checkpoint_paths)
                    update_overlay(f"Loading {play_env.checkpoint_paths[checkpoint_index].name}...")
                    viewer.sync()
                    try:
                        play_env.load_checkpoint(checkpoint_index)
                    except Exception as exception:
                        status = "Load failed; keeping current eval (see terminal)"
                        print(f"Could not load checkpoint: {exception}")
                    else:
                        rng = reset_spawn()
                        status = "Same spawn restarted"
                elif keycode == glfw.KEY_DOWN:
                    rng = reset_spawn()
                    status = "Same spawn restarted"
                elif keycode == glfw.KEY_UP:
                    spawn_seed += 1
                    rng = reset_spawn()
                    status = "New spawn"

            # Advance the simulation
            if not paused:
                _, _, rng, done = play_env.step(model, data, rng)
                if done:
                    spawn_seed += 1
                    rng = reset_spawn()
                    status = "New episode"
                else:
                    mujoco.mj_forward(model, data)

            # Refresh the display and pace playback
            update_overlay(status)
            viewer.sync()

            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, play_env.ctrl_dt - elapsed))


def main(argv):
    del argv
    play_env = PLAY_ENV_REGISTRY["sumo"](run_dir=_RUN_DIR.value, eval_number=_EVAL.value)
    run_viewer(play_env, seed=_SEED.value)


if __name__ == "__main__":
    app.run(main)
