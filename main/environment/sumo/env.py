from typing import Any, Optional, Union
from mujoco_playground._src import mjx_env
from mujoco_playground import locomotion
from jax import numpy as jp
import jax
import mujoco
from mujoco import mjx
from ml_collections import config_dict

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        iterations=50,
        ls_iterations=10,
        naconmax=150000,
        njmax=200,
        seed=0,
        episode_length=1000,
    )

class Sumo(mjx_env.MjxEnv):
    """ Environment to two humanoid to play sumo"""


    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict[str, Union[str, int, list[Any]]]] = None,
    ):

        super().__init__(config, config_overrides)
        self._xml_path  = "main/environment/sumo/model.xml"
        self._mj_model  = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mj_model.opt.iterations = self._config.iterations
        self._mj_model.opt.ls_iterations = self._config.ls_iterations
        self._mjx_model = mjx_env.put_model(self._mj_model, impl=self._config.impl)

        # Environment constants
        # action scale formula: 1.0 * effort_limit / stiffness
        self._action_scale = jp.array(
            1.0 * self._mj_model.actuator_forcerange[:, 1]
            / self._mj_model.actuator_gainprm[:, 0]
        )
        self._standing_height          = 1.0
        self._standing_reward          = 5.0
        self._terminal_reward          = 1000.0
        self._draw_reward              = -1000.0
        self._episode_length           = self._config.episode_length
        self._min_height              = 0.5
        self._min_upright_alignment   = jp.cos(jp.deg2rad(70.0))
        self._blue_humanoid_site_index = self._mj_model.site("blue_humanoid").id
        self._red_humanoid_site_index  = self._mj_model.site("red_humanoid").id

        self._start_pose               = jp.array(self._mj_model.keyframe("home").qpos)
        self._radius = 2


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
        for joint_id in range(self._mj_model.njnt):
            joint_name = self._mj_model.joint(joint_id).name

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
        for body_id in range(self._mj_model.nbody):
            body_name = self._mj_model.body(body_id).name

            # Store the body indices of humanoids (for Observation Space)
            if "red" in body_name and "humanoid" in body_name:
                red_humanoid_body_ids.append(body_id)

            if "blue" in body_name and "humanoid" in body_name:
                blue_humanoid_body_ids.append(body_id)

        # Store actuator IDs separately from joint/DOF IDs.
        for actuator_id in range(self._mj_model.nu):
            actuator_name = self._mj_model.actuator(actuator_id).name

            if "red" in actuator_name:
                red_humanoid_actuator_ids.append(actuator_id)

            if "blue" in actuator_name:
                blue_humanoid_actuator_ids.append(actuator_id)

        # Collect raycast and contact-force sensor data indices for each humanoid.
        for sensor_id in range(self._mj_model.nsensor):
            sensor_name = self._mj_model.sensor(sensor_id).name
            sensor_type = self._mj_model.sensor_type[sensor_id]
            adr = self._mj_model.sensor_adr[sensor_id]
            dim = self._mj_model.sensor_dim[sensor_id]
            data_indices = range(adr, adr + dim)

            is_rangefinder = (
                sensor_type == mujoco.mjtSensor.mjSENS_RANGEFINDER
            )
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

        actuator_qpos_ids = self._mj_model.jnt_qposadr[self._mj_model.actuator_trnid[:, 0]]
        self._home_joint_targets = self._start_pose[actuator_qpos_ids]

    # Resets the environment to an initial state
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2, rng3, rng4, rgn5 = jax.random.split(rng, 6)

        # Sample independent positions, rejecting when humanoids are less than 0.5 m apart
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
        qvel = jp.zeros((self._mj_model.nv,))

        data = mjx_env.make_data(
            self.mj_model,
            qpos     = qpos,
            qvel     = qvel,
            impl     = self.mjx_model.impl.value,
            njmax    = self._config.njmax,
            naconmax = self._config.naconmax,
        )
        data = mjx.forward(self.mjx_model, data)

        reward, done, zero = jp.zeros(3)
        info    = {
            "rng": rng,
            "steps": jp.array(0.0),
            "red_humanoid_last_action": jp.zeros(len(self._red_humanoid_actuator_ids)),
            "blue_humanoid_last_action": jp.zeros(len(self._blue_humanoid_actuator_ids)),
            "agent_rewards": {
                "red": jp.array(0.0),
                "blue": jp.array(0.0)
            },
            "agent_terminated": {
                "red": jp.array(False),
                "blue": jp.array(False),
            },
            "contact": jp.array(False),
            "draw": jp.array(False),
        }
        obs = self._calculate_obs(data, info)
        metrics = {"velocity": zero}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, actions: dict[str, jax.Array]) -> mjx_env.State:
        data0  = state.data
        red_humanoid_actions = actions["red"]
        blue_humanoid_actions = actions["blue"]
        ctrl = self._home_joint_targets + self._action_scale * jp.concatenate([red_humanoid_actions,blue_humanoid_actions], axis=-1)

        # Check every physics substep so brief humanoid-to-humanoid contact counts.
        def physics_step(data, _):
            data = mjx.step(self.mjx_model, data.replace(ctrl=ctrl))
            contact = mjx_env.get_sensor_data(
                self._mj_model, data, "humanoid_contact"
            )[0] > 0
            return data, contact

        data, contacts = jax.lax.scan(
            physics_step, data0, None, length=self.n_substeps
        )
        state.info["contact"] = jp.any(contacts)

        termination = self._calculate_termination(data)
        state.info["draw"] = (state.info["steps"] + 1 >= self._episode_length) & (termination == 0)

        rewards = self._calculate_reward(data, actions, state.info)
        state.info["red_humanoid_last_action"] = red_humanoid_actions
        state.info["blue_humanoid_last_action"] = blue_humanoid_actions

        # Observe the completed step without incrementing the wrapper's counter
        obs_info = {**state.info, "steps": state.info["steps"] + 1}
        obs    = self._calculate_obs(data, obs_info)
        state.info["agent_rewards"] = rewards
        # Preserve the terminal outcome before the wrapper restores reset data
        state.info["agent_terminated"] = self._calculate_agent_terminations(data)
        # Ending the match clock is terminal, so EpisodeWrapper won't mark it
        # as a truncation and GAE will retain the draw penalty without bootstrap
        termination = jp.where(state.info["draw"], 1.0, termination)
        return mjx_env.State(data, obs, jp.array(0.0), termination, state.metrics, state.info)

    def _calculate_agent_terminations(self, data: mjx.Data) -> dict[str, jax.Array]:
        blue_humanoid_dist = jp.linalg.norm(data.qpos[self._blue_humanoid_qpos_ids[0:2]])
        red_humanoid_dist = jp.linalg.norm(data.qpos[self._red_humanoid_qpos_ids[0:2]])
        
        # Terminate if either humanoid's torso drops below the minimum height.
        blue_humanoid_torso_height = data.site_xpos[self._blue_humanoid_site_index][2]
        red_humanoid_torso_height = data.site_xpos[self._red_humanoid_site_index][2]

        # Terminate if either humanoid's torso tilts to far of z axis
        blue_humanoid_up = data.site_xmat[self._blue_humanoid_site_index][2, 2]
        red_humanoid_up = data.site_xmat[self._red_humanoid_site_index][2, 2]
        blue_humanoid_fallen = jp.logical_or(
            blue_humanoid_torso_height < self._min_height,
            blue_humanoid_up < self._min_upright_alignment,
        )
        red_humanoid_fallen = jp.logical_or(
            red_humanoid_torso_height < self._min_height,
            red_humanoid_up < self._min_upright_alignment,
        )

        return {
            "red": jp.logical_or(red_humanoid_fallen, red_humanoid_dist > self._radius),
            "blue": jp.logical_or(blue_humanoid_fallen, blue_humanoid_dist > self._radius),
        }

    def _calculate_termination(self, data: mjx.Data) -> jax.Array:
        agent_terminated = self._calculate_agent_terminations(data)
        reset_condition = jp.logical_or(agent_terminated["red"], agent_terminated["blue"])
        return jp.where(reset_condition, 1.0, 0.0)

    def _calculate_reward(
        self,
        data: mjx.Data,
        actions: dict[str, jax.Array],
        info: dict[str, Any],
    ) -> dict[str, jax.Array]:
        # Prefer standing tall without terminating ordinary crouching (from openAI paper)
        red_standing_reward = jp.where(
            data.site_xpos[self._red_humanoid_site_index][2] >= self._standing_height,
            self._standing_reward,
            -self._standing_reward,
        )
        blue_standing_reward = jp.where(
            data.site_xpos[self._blue_humanoid_site_index][2] >= self._standing_height,
            self._standing_reward,
            -self._standing_reward,
        )
        red_humanoid_dist = jp.linalg.norm(data.qpos[self._red_humanoid_qpos_ids[:2]])
        blue_humanoid_dist = jp.linalg.norm(data.qpos[self._blue_humanoid_qpos_ids[:2]])
        agent_terminated = self._calculate_agent_terminations(data)

        # red_edge_reward = blue_humanoid_dist / self._radius
        # blue_edge_reward = red_humanoid_dist / self._radius

        # Reward holding the centre 
        center_reward_weight = 0.5
        red_center_reward = jp.clip(1.0 - red_humanoid_dist / self._radius, 0.0, 1.0)
        blue_center_reward = jp.clip(1.0 - blue_humanoid_dist / self._radius, 0.0, 1.0)

        # Big reward/penalty for losing the event/match
        red_terminal_reward = jp.where(
            agent_terminated["red"], -self._terminal_reward,
            jp.where(agent_terminated["blue"], self._terminal_reward, 0.0),
        )
        blue_terminal_reward = jp.where(
            agent_terminated["blue"], -self._terminal_reward,
            jp.where(agent_terminated["red"], self._terminal_reward, 0.0),
        )
        draw_reward = jp.where(info["draw"], self._draw_reward, 0.0)
        return {
            "red": (
                red_standing_reward
                # + red_edge_reward
                + center_reward_weight * red_center_reward
            ) * self.dt + red_terminal_reward + draw_reward,
            "blue": (
                blue_standing_reward
                # + blue_edge_reward
                + center_reward_weight * blue_center_reward
            ) * self.dt + blue_terminal_reward + draw_reward,
        }

    def _calculate_obs(
        self, data: mjx.Data, info: dict[str, jax.Array]
    ) -> mjx_env.Observation:

        # Get the local base linear velocity of both humanoids
        red_humanoid_linvel  = mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_local_linvel")
        blue_humanoid_linvel = mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_local_linvel")

        # Get the local base angular velocity of both humanoids
        red_humanoid_angvel  = mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_gyro")
        blue_humanoid_angvel = mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_gyro")

        # Relative joint positions for both humanoid
        red_humanoid_joint_positions  = data.qpos[self._red_humanoid_qpos_ids[7:]] - self._start_pose[self._red_humanoid_qpos_ids[7:]]
        blue_humanoid_joint_positions = data.qpos[self._blue_humanoid_qpos_ids[7:]] - self._start_pose[self._blue_humanoid_qpos_ids[7:]]

        # Joint velocities for both humanoid
        red_humanoid_joint_velocities = data.qvel[self._red_humanoid_qvel_ids[6:]]
        blue_humanoid_joint_velocities = data.qvel[self._blue_humanoid_qvel_ids[6:]]

        # projected_gravity for each humanoid
        red_humanoid_gravity = data.site_xmat[self._red_humanoid_site_index].T @ jp.array([0, 0, -1])
        blue_humanoid_gravity = data.site_xmat[self._blue_humanoid_site_index].T @ jp.array([0, 0, -1])

        red_xy = data.qpos[self._red_humanoid_qpos_ids[:2]]
        blue_xy = data.qpos[self._blue_humanoid_qpos_ids[:2]]

        red_humanoid_ring_edge_distance = jp.atleast_1d(
            1.0 - jp.linalg.norm(red_xy) / self._radius
        )
        blue_humanoid_ring_edge_distance = jp.atleast_1d(
            1.0 - jp.linalg.norm(blue_xy) / self._radius
        )

        def local_xy(world_vector, site_index):
            rotation = data.site_xmat[site_index]
            heading = jp.arctan2(rotation[1, 0], rotation[0, 0])
            cos_heading, sin_heading = jp.cos(heading), jp.sin(heading)
            return jp.array([
                cos_heading * world_vector[0] + sin_heading * world_vector[1],
                -sin_heading * world_vector[0] + cos_heading * world_vector[1],
            ]) / self._radius

        red_humanoid_to_center = local_xy(-red_xy, self._red_humanoid_site_index)
        blue_humanoid_to_center = local_xy(-blue_xy, self._blue_humanoid_site_index)
        red_humanoid_to_opponent = local_xy(blue_xy - red_xy, self._red_humanoid_site_index)
        blue_humanoid_to_opponent = local_xy(red_xy - blue_xy, self._blue_humanoid_site_index)

        # Foot contact and force observations
        blue_humanoid_foot_contact = jp.array([
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_left_foot_floor_contact")[0] > 0,
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_right_foot_floor_contact")[0] > 0,
        ])

        red_humanoid_foot_contact = jp.array([
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_left_foot_floor_contact")[0] > 0,
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_right_foot_floor_contact")[0] > 0,
        ])


        red_humanoid_foot_forces = jp.concatenate([
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_left_foot_contact_force"),
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_right_foot_contact_force"),
        ])

        blue_humanoid_foot_forces = jp.concatenate([
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_left_foot_contact_force"),
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_right_foot_contact_force"),
        ])

        red_humanoid_last_action = info["red_humanoid_last_action"]
        blue_humanoid_last_action = info["blue_humanoid_last_action"]

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

        # Match clock
        time_remaining = jp.atleast_1d(jp.clip(
            1.0 - info["steps"] / self._episode_length, 0.0, 1.0
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

        global_qpos = data.qpos
        global_qvel = data.qvel
        relative_position = (
            global_qpos[self._blue_humanoid_qpos_ids[:3]]
            - global_qpos[self._red_humanoid_qpos_ids[:3]]
        )
        relative_distance = jp.linalg.norm(relative_position, keepdims=True)
        contact_forces = jp.concatenate([
            data.sensordata[self._red_humanoid_contact_force_ids],
            data.sensordata[self._blue_humanoid_contact_force_ids],
        ])
        global_critic_obs = jp.concatenate([
            red_humanoid_obs,
            red_humanoid_to_center,
            red_humanoid_to_opponent,
            blue_humanoid_obs,
            blue_humanoid_to_center,
            blue_humanoid_to_opponent,
            global_qpos,
            global_qvel,
            relative_position,
            relative_distance,
            contact_forces,
        ])

        return {
            "actor": {
                "red": red_humanoid_obs,
                "blue": blue_humanoid_obs,
            },
            "critic" : global_critic_obs,
        }

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

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def observation_size(self) -> dict[str, int]:
        return {
            "actor": 134,
            "critic": 476,
        }

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

locomotion.register_environment('sumo', Sumo, default_config)
