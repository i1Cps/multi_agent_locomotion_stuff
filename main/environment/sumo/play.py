from pathlib import Path
import json

from main.environment.sumo.env import default_config
from main.multi_agent_algorithm.network import ActorNetwork
import mujoco
from orbax import checkpoint as ocp
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

# Environment to help the inference engine construct observation and evaluate the state for a control policy
class Sumo_Play:
    def __init__(self, run_dir: str | Path | None = None, eval_number: int | None = None):
        self._xml_path  = "main/environment/sumo/model.xml"
        self.mj_model  = mujoco.MjModel.from_xml_path(self._xml_path)
        self.sim_dt     = 0.005
        self.ctrl_dt    = 0.02
        self.mj_model.opt.timestep = self.sim_dt
        config = default_config()
        self.mj_model.opt.iterations = config.iterations
        self.mj_model.opt.ls_iterations = config.ls_iterations
        self.skipped_frames = self.ctrl_dt / self.sim_dt
        self._agent_action_size = 17
        with open("main/environment/sumo/options.json") as f:
            options = json.load(f)
        self.episode_length = options["ppo"]["episode_length"]
        self.episode_steps = 0

        # Find the run's evaluation checkpoints and load the selected pair of policies
        self.checkpoint_paths = self.find_checkpoints(run_dir)
        checkpoint_index = len(self.checkpoint_paths) - 1
        if eval_number is not None:
            eval_numbers = [int(path.name.removeprefix("eval_")) for path in self.checkpoint_paths]
            if eval_number not in eval_numbers:
                raise ValueError(f"Eval {eval_number} not found. Available evals: {eval_numbers}")
            checkpoint_index = eval_numbers.index(eval_number)
        self._actor_network = ActorNetwork((512, 256, 128), self._agent_action_size)
        self.load_checkpoint(checkpoint_index)

        # Environment constants
        # Same per joint target scales as training, derived from the compiled XML
        self._action_scale = jp.array(
            1.0 * self.mj_model.actuator_forcerange[:, 1]
            / self.mj_model.actuator_gainprm[:, 0]
        )
        self._reset_noise_scale    = 0.01
        self._agent_healthy_height = 0.5
        self._red_humanoid_site_id = self.mj_model.site("red_humanoid").id
        self._blue_humanoid_site_id = self.mj_model.site("blue_humanoid").id
        self._red_humanoid_last_action = jp.zeros(self._agent_action_size)
        self._blue_humanoid_last_action = jp.zeros(self._agent_action_size)

        self._radius = 2
        self._max_tilt = jp.cos(jp.deg2rad(70.0))
        self._min_height = 0.3
        self._ticks_before_reset = 100

        # Collect humanoid id's
        red_humanoid_qpos_ids = []
        red_humanoid_qvel_ids = []
        red_humanoid_body_ids = []
        red_humanoid_raycast_ids = []
        red_humanoid_contact_force_ids = []
        red_humanoid_actuator_ids = []

        blue_humanoid_qpos_ids = []
        blue_humanoid_qvel_ids = []
        blue_humanoid_body_ids = []
        blue_humanoid_raycast_ids = []
        blue_humanoid_contact_force_ids = []
        blue_humanoid_actuator_ids = []

        curr_qpos_id = 0
        curr_qvel_id = 0

        # Loop through each joint in the world model
        for joint_id in range(self.mj_model.njnt):
            joint_name = self.mj_model.joint(joint_id).name

            # Check if joint is a root joint
            if "root" in joint_name:
                # Add joint id to correct humanoid id list
                if "red" in joint_name:
                    red_humanoid_qpos_ids.extend(range(curr_qpos_id, curr_qpos_id + 7))
                    red_humanoid_qvel_ids.extend(range(curr_qvel_id, curr_qvel_id + 6))
                if "blue" in joint_name:
                    blue_humanoid_qpos_ids.extend(range(curr_qpos_id, curr_qpos_id + 7))
                    blue_humanoid_qvel_ids.extend(range(curr_qvel_id, curr_qvel_id + 6))

                # Increment the index counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_id += 7
                curr_qvel_id += 6

            # Handle non root joints
            else:
                if "red" in joint_name:
                    red_humanoid_qpos_ids.append(curr_qpos_id)
                    red_humanoid_qvel_ids.append(curr_qvel_id)

                if "blue" in joint_name:
                    blue_humanoid_qpos_ids.append(curr_qpos_id)
                    blue_humanoid_qvel_ids.append(curr_qvel_id)
                curr_qpos_id += 1
                curr_qvel_id += 1

        # Loop through each body in the world model
        for body_id in range(self.mj_model.nbody):
            body_name = self.mj_model.body(body_id).name

            # Store the body indices of humanoids (for Observation Space)
            if "red" in body_name and "humanoid" in body_name:
                red_humanoid_body_ids.append(body_id)

            if "blue" in body_name and "humanoid" in body_name:
                blue_humanoid_body_ids.append(body_id)

        for actuator_id in range(self.mj_model.nu):
            actuator_name = self.mj_model.actuator(actuator_id).name

            if "red" in actuator_name:
                red_humanoid_actuator_ids.append(actuator_id)

            if "blue" in actuator_name:
                blue_humanoid_actuator_ids.append(actuator_id)

        # Map home joint angles into actuator order, which differs from qpos order
        actuator_qpos_ids = self.mj_model.jnt_qposadr[
            self.mj_model.actuator_trnid[:, 0]
        ]

        # Collect raycast and contact force sensor data indices for each humanoid
        for sensor_id in range(self.mj_model.nsensor):
            sensor_name = self.mj_model.sensor(sensor_id).name
            sensor_type = self.mj_model.sensor_type[sensor_id]
            adr = self.mj_model.sensor_adr[sensor_id]
            dim = self.mj_model.sensor_dim[sensor_id]
            data_indices = range(adr, adr + dim)

            is_rangefinder = (sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER)
            if "ray" in sensor_name and is_rangefinder:
                if "red" in sensor_name:
                    red_humanoid_raycast_ids.extend(data_indices)

                if "blue" in sensor_name:
                    blue_humanoid_raycast_ids.extend(data_indices)

            is_contact_force = (
                sensor_type == mujoco.mjtSensor.mjSENS_CONTACT
                and sensor_name.endswith("_contact_force")
            )
            if is_contact_force:
                if dim != 3:
                    raise ValueError(
                        f"Expected a 3D contact force from {sensor_name}, got {dim}."
                    )

                if "red" in sensor_name:
                    red_humanoid_contact_force_ids.extend(data_indices)

                if "blue" in sensor_name:
                    blue_humanoid_contact_force_ids.extend(data_indices)

        # Convert the lists to JAX arrays
        self._red_humanoid_qpos_ids = jp.array(red_humanoid_qpos_ids)
        self._red_humanoid_qvel_ids = jp.array(red_humanoid_qvel_ids)
        self._red_humanoid_body_ids = jp.array(red_humanoid_body_ids)
        self._red_humanoid_raycast_ids  = jp.array(red_humanoid_raycast_ids)
        self._red_humanoid_contact_force_ids = jp.array(red_humanoid_contact_force_ids)
        self._red_humanoid_actuator_ids = jp.array(red_humanoid_actuator_ids)

        self._blue_humanoid_qpos_ids = jp.array(blue_humanoid_qpos_ids)
        self._blue_humanoid_qvel_ids = jp.array(blue_humanoid_qvel_ids)
        self._blue_humanoid_body_ids = jp.array(blue_humanoid_body_ids)
        self._blue_humanoid_raycast_ids  = jp.array(blue_humanoid_raycast_ids)
        self._blue_humanoid_contact_force_ids = jp.array(blue_humanoid_contact_force_ids)
        self._blue_humanoid_actuator_ids = jp.array(blue_humanoid_actuator_ids)

        self._start_pose = jp.array(self.mj_model.keyframe("home").qpos)
        self._home_joint_targets = self._start_pose[actuator_qpos_ids]

        self._red_unhealthy_ticks = jp.array(1)
        self._blue_unhealthy_ticks = jp.array(1)

    @staticmethod
    def find_checkpoints(run_dir: str | Path | None) -> list[Path]:
        """Use a selected run, or the newest run containing completed eval saves"""
        if run_dir is None:
            runs = sorted(Path("main/models/sumo").glob("sumo-*"), reverse=True)
        else:
            runs = [Path(run_dir)]
        for run in runs:
            checkpoints = [
                path.resolve() for path in run.glob("eval_*")
                if path.name.removeprefix("eval_").isdigit()
                and (path / "_CHECKPOINT_METADATA").is_file()
            ]
            if checkpoints:
                return sorted(checkpoints, key=lambda path: int(path.name.removeprefix("eval_")))
        raise FileNotFoundError(f"No completed eval checkpoints in {run_dir or 'main/models/sumo/sumo-*'}")

    @property
    def checkpoint_path(self) -> Path:
        return self.checkpoint_paths[self.checkpoint_index]

    def load_checkpoint(self, index: int):
        """Replace both actors and their normalisers only after a successful restore"""
        path = self.checkpoint_paths[index]
        with ocp.StandardCheckpointer() as checkpointer:
            checkpoint = checkpointer.restore(path)
        actors = checkpoint["actor_params"]
        normalisers = checkpoint["actor_normaliser_params"]
        action_log_std = checkpoint["action_log_std"]
        self._actor_params = actors
        self._actor_normaliser_params = normalisers
        self._action_log_std = action_log_std
        self.checkpoint_index = index
        print(f"Loaded {path.parent.name}/{path.name} ({index + 1}/{len(self.checkpoint_paths)})")

    # Resets the environment to an initial state
    def reset(self, data: mujoco.MjData, rng: jax.Array) -> tuple[mujoco.MjData, jax.Array]:
        # Clear controls, forces and previous actions as well as the pose so that
        # replaying a seed gives the same start after switching checkpoints.
        mujoco.mj_resetData(self.mj_model, data)
        self.episode_steps = 0
        self._red_unhealthy_ticks = jp.array(1)
        self._blue_unhealthy_ticks = jp.array(1)
        self._red_humanoid_last_action = jp.zeros(self._agent_action_size)
        self._blue_humanoid_last_action = jp.zeros(self._agent_action_size)
        rng, rng1, rng2, rng3, rng4, rgn5 = jax.random.split(rng, 6)

        # Sample independent positions
        _, sampled_radius, sampled_theta = self.sample_positions_in_circle(rng1, radius=self._radius)
        x1 = sampled_radius[0] * jp.cos(sampled_theta[0])
        y1 = sampled_radius[0] * jp.sin(sampled_theta[0])
        x2 = sampled_radius[1] * jp.cos(sampled_theta[1])
        y2 = sampled_radius[1] * jp.sin(sampled_theta[1])
        z  = self._start_pose[2]

        # Face the arena centre from each sampled root position
        azimuth = jp.array([jp.arctan2(-y1, -x1), jp.arctan2(-y2, -x2)])
        red_orientation  = jp.array([jp.cos(azimuth[0] / 2), 0, 0, jp.sin(azimuth[0] / 2)])
        blue_orientation = jp.array([jp.cos(azimuth[1] / 2), 0, 0, jp.sin(azimuth[1] / 2)])

        red_humanoid_pose  = jp.concatenate([jp.array([x1, y1, z]), red_orientation])
        blue_humanoid_pose = jp.concatenate([jp.array([x2, y2, z]), blue_orientation])

        red_humanoid_joint_positions  = self._start_pose[self._red_humanoid_qpos_ids][7:]
        blue_humanoid_joint_positions = self._start_pose[self._blue_humanoid_qpos_ids][7:]

        red_humanoid_qpos  = jp.concatenate([red_humanoid_pose, red_humanoid_joint_positions])
        blue_humanoid_qpos = jp.concatenate([blue_humanoid_pose, blue_humanoid_joint_positions])

        qpos = jp.concatenate([red_humanoid_qpos, blue_humanoid_qpos])
        qvel = jp.zeros((self.mj_model.nv,))

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.qacc_warmstart[:] = 0.0
        data.qfrc_actuator[:] = 0.0
        data.ctrl[:] = self._home_joint_targets
        return data, rng

    def step(self, model, data:mujoco.MjData, rng):
        next_rng, obs_rng, rng = jax.random.split(rng, 3)

        mjx_data = mjx.put_data(model, data, impl="jax")
        done = self._calculate_termination(mjx_data)
        red_humanoid_obs, blue_humanoid_obs = self._calculate_obs(mjx_data)

        # Get actions
        red_humanoid_actions, blue_humanoid_actions = self.calculate_actions(red_humanoid_obs, blue_humanoid_obs)
        actions = jp.concatenate([red_humanoid_actions, blue_humanoid_actions])
        data.ctrl = self._home_joint_targets + self._action_scale * actions

        # Step the environment
        for frame in range(int(self.skipped_frames)):
            mujoco.mj_step(model, data)

        self.episode_steps += 1
        # The viewer handles resets
        done = bool(done or self.episode_steps >= self.episode_length)
        return model, data, next_rng, done



    def _calculate_termination(self, data: mjx.Data):
        # Reset when either humanoid's root leaves the ring
        blue_humanoid_dist = jp.linalg.norm(data.qpos[self._blue_humanoid_qpos_ids[0:2]])
        red_humanoid_dist = jp.linalg.norm(data.qpos[self._red_humanoid_qpos_ids[0:2]])
        humanoid_out = jp.logical_or(blue_humanoid_dist > self._radius, red_humanoid_dist > self._radius)

        # Reset the episode if one of the agents falls over, over a peirod of time (ticks)
        # Terminate if one of the bodies rotates to far
        blue_humanoid_torso_height = data.site_xpos[self._blue_humanoid_site_id][2]
        red_humanoid_torso_height = data.site_xpos[self._red_humanoid_site_id][2]

        blue_humanoid_fallen = blue_humanoid_torso_height < self._min_height
        red_humanoid_fallen = red_humanoid_torso_height < self._min_height

        # Use a tick system to keep track of how long the agent has been in an unhealthy state for
        new_red_unhealthy_ticks = jp.where(red_humanoid_fallen, self._red_unhealthy_ticks + 1, self._red_unhealthy_ticks)
        self._red_unhealthy_ticks = new_red_unhealthy_ticks

        new_blue_unhealthy_ticks = jp.where(blue_humanoid_fallen, self._blue_unhealthy_ticks + 1, self._blue_unhealthy_ticks)
        self._blue_unhealthy_ticks = new_blue_unhealthy_ticks

        reset_due_to_unhealthy = jp.logical_or(self._blue_unhealthy_ticks > self._ticks_before_reset , self._red_unhealthy_ticks > self._ticks_before_reset)

        reset = jp.logical_or(reset_due_to_unhealthy, humanoid_out)

        self._red_unhealthy_ticks = jp.where(reset, 1, self._red_unhealthy_ticks)
        self._blue_unhealthy_ticks = jp.where(reset, 1, self._blue_unhealthy_ticks)

        return reset

    def _calculate_obs(self, data: mjx.Data) -> tuple[jp.ndarray, jp.ndarray]:
        # Get the local base linear velocity of both humanoids
        red_humanoid_linvel  = mjx_env.get_sensor_data(self.mj_model, data, "red_humanoid_local_linvel")
        blue_humanoid_linvel = mjx_env.get_sensor_data(self.mj_model, data, "blue_humanoid_local_linvel")

        # Get the local base angular velocity of both humanoids
        red_humanoid_angvel  = mjx_env.get_sensor_data(self.mj_model, data, "red_humanoid_gyro")
        blue_humanoid_angvel = mjx_env.get_sensor_data(self.mj_model, data, "blue_humanoid_gyro")

        # Relative joint positions for both humanoid
        red_humanoid_joint_positions  = data.qpos[self._red_humanoid_qpos_ids[7:]] - self._start_pose[self._red_humanoid_qpos_ids[7:]]
        blue_humanoid_joint_positions = data.qpos[self._blue_humanoid_qpos_ids[7:]] - self._start_pose[self._blue_humanoid_qpos_ids[7:]]

        # Joint velocities for both humanoid
        red_humanoid_joint_velocities = data.qvel[self._red_humanoid_qvel_ids[6:]]
        blue_humanoid_joint_velocities = data.qvel[self._blue_humanoid_qvel_ids[6:]]

        # projected_gravity for each humanoid
        red_humanoid_gravity = data.site_xmat[self._red_humanoid_site_id].T @ jp.array([0, 0, -1])
        blue_humanoid_gravity = data.site_xmat[self._blue_humanoid_site_id].T @ jp.array([0, 0, -1])

        # Same normalized radial clearance as the training actors; no direction.
        red_humanoid_ring_edge_distance = jp.atleast_1d(
            1.0 - jp.linalg.norm(data.qpos[self._red_humanoid_qpos_ids[:2]]) / self._radius
        )
        blue_humanoid_ring_edge_distance = jp.atleast_1d(
            1.0 - jp.linalg.norm(data.qpos[self._blue_humanoid_qpos_ids[:2]]) / self._radius
        )

        # Foot contact and force observations.
        blue_humanoid_foot_contact = jp.array([
            mjx_env.get_sensor_data(self.mj_model, data, "blue_humanoid_left_foot_floor_contact")[0] > 0,
            mjx_env.get_sensor_data(self.mj_model, data, "blue_humanoid_right_foot_floor_contact")[0] > 0,
        ])

        red_humanoid_foot_contact = jp.array([
            mjx_env.get_sensor_data(self.mj_model, data, "red_humanoid_left_foot_floor_contact")[0] > 0,
            mjx_env.get_sensor_data(self.mj_model, data, "red_humanoid_right_foot_floor_contact")[0] > 0,
        ])


        red_humanoid_foot_forces = jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, "red_humanoid_left_foot_contact_force"),
            mjx_env.get_sensor_data(self.mj_model, data, "red_humanoid_right_foot_contact_force"),
        ])

        blue_humanoid_foot_forces = jp.concatenate([
            mjx_env.get_sensor_data(self.mj_model, data, "blue_humanoid_left_foot_contact_force"),
            mjx_env.get_sensor_data(self.mj_model, data, "blue_humanoid_right_foot_contact_force"),
        ])

        red_humanoid_last_action = self._red_humanoid_last_action
        blue_humanoid_last_action = self._blue_humanoid_last_action

        # red_humanoid_last_action = info["red_humanoid_last_action"]
        # blue_humanoid_last_action = info["blue_humanoid_last_action"]

        red_humanoid_raycast_distances = data.sensordata[
            self._red_humanoid_raycast_ids
        ]
        blue_humanoid_raycast_distances = data.sensordata[
            self._blue_humanoid_raycast_ids
        ]
        no_intersection = -1.0

        red_humanoid_raycast_obs = jp.where(
            red_humanoid_raycast_distances == no_intersection,
            1.0,
            jp.tanh(red_humanoid_raycast_distances),
        )
        blue_humanoid_raycast_obs = jp.where(
            blue_humanoid_raycast_distances == no_intersection,
            1.0,
            jp.tanh(blue_humanoid_raycast_distances),
        )

        # Same normalized match clock as the training actors.
        time_remaining = jp.atleast_1d(jp.clip(
            1.0 - self.episode_steps / self.episode_length, 0.0, 1.0
        ))

        red_humanoid_obs = jp.concatenate([
            red_humanoid_linvel,
            red_humanoid_angvel,
            red_humanoid_gravity,
            red_humanoid_joint_positions,
            red_humanoid_joint_velocities,
            red_humanoid_foot_contact,
            red_humanoid_foot_forces,
            red_humanoid_last_action,
            red_humanoid_raycast_obs,
            red_humanoid_ring_edge_distance,
            time_remaining,
        ])
        blue_humanoid_obs = jp.concatenate([
            blue_humanoid_linvel,
            blue_humanoid_angvel,
            blue_humanoid_gravity,
            blue_humanoid_joint_positions,
            blue_humanoid_joint_velocities,
            blue_humanoid_foot_contact,
            blue_humanoid_foot_forces,
            blue_humanoid_last_action,
            blue_humanoid_raycast_obs,
            blue_humanoid_ring_edge_distance,
            time_remaining,
        ])

        return red_humanoid_obs, blue_humanoid_obs

    def calculate_actions(self, red_humanoid_obs, blue_humanoid_obs):
        # Normalise the observations
        red_normaliser_stats = self._actor_normaliser_params["red"]
        blue_normaliser_stats = self._actor_normaliser_params["blue"]

        red_humanoid_obs = (red_humanoid_obs - red_normaliser_stats["mean"])/ jp.sqrt(red_normaliser_stats["var"] + 1e-8)
        blue_humanoid_obs = (blue_humanoid_obs - blue_normaliser_stats["mean"])/ jp.sqrt(blue_normaliser_stats["var"] + 1e-8)

        red_network_params = self._actor_params["red"]
        blue_network_params = self._actor_params["blue"]

        red_mean_action = self._actor_network.apply(red_network_params, red_humanoid_obs)
        blue_mean_action = self._actor_network.apply(blue_network_params, blue_humanoid_obs)

        red_action = jp.tanh(red_mean_action)
        blue_action = jp.tanh(blue_mean_action)

        self._red_humanoid_last_action = red_action
        self._blue_humanoid_last_action = blue_action

        return red_action, blue_action


    def sample_positions_in_circle(self, key, radius = 2, too_close = 0.5):
        def sample(args):
            key, _, _ = args
            key, radius_key, theta_key = jax.random.split(key, 3)
            sampled_radius = jax.random.uniform(radius_key, (2,), minval=0, maxval=radius)
            sampled_theta = jax.random.uniform(theta_key, (2,), minval=-jp.pi, maxval=jp.pi)
            return key, sampled_radius, sampled_theta

        def collision(args):
            _, sampled_radius, sampled_theta = args
            x = sampled_radius * jp.cos(sampled_theta)
            y = sampled_radius * jp.sin(sampled_theta)
            distance = jp.sqrt((x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2)
            return distance < too_close

        first_sample = sample((key, jp.zeros(2), jp.zeros(2)))
        return jax.lax.while_loop(collision, sample, first_sample)
