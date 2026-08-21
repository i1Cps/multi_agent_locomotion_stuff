from typing import Any, Dict, Optional, Union
from mujoco_playground._src import mjx_env
from mujoco_playground import locomotion
from jax import numpy as jp
import jax
import mujoco
from mujoco import mjx
from ml_collections import config_dict

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        # Sim options are set from options.json
        naconmax=90000,
        njmax=200,
        seed=0,
    )

class Sumo(mjx_env.MjxEnv):
    """ Environment to two humanoid to play sumo"""


    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):

        super().__init__(config, config_overrides)
        self._xml_path  = "main/environment/sumo/model.xml"
        self._mj_model  = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Environment constants
        self._agent_healthy_height     = 1.00
        self._blue_humanoid_site_index = self._mj_model.site("blue_humanoid").id
        self._red_humanoid_site_index  = self._mj_model.site("red_humanoid").id

        self._start_pose          = self._mj_model.keyframe("home").qpos
        self._radius = 5
        self._max_tilt = jp.cos(jp.deg2rad(70.0))


        # Collect humanoid id's
        red_humanoid_qpos_ids = []
        red_humanoid_qvel_ids = []
        red_humanoid_body_ids = []
        red_humanoid_raycast_ids = []
        red_humanoid_actuator_ids = []

        blue_humanoid_qpos_ids = []
        blue_humanoid_qvel_ids = []
        blue_humanoid_body_ids = []
        blue_humanoid_raycast_ids = []
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

        # Collect raycast sensor data indices for each humanoid.
        for sensor_id in range(self._mj_model.nsensor):
            sensor_name = self._mj_model.sensor(sensor_id).name
            is_rangefinder = (
                self._mj_model.sensor_type[sensor_id] == mujoco.mjtSensor.mjSENS_RANGEFINDER
            )
            if "ray" not in sensor_name or not is_rangefinder:
                continue

            adr = self._mj_model.sensor_adr[sensor_id]
            dim = self._mj_model.sensor_dim[sensor_id]
            data_indices = range(adr, adr + dim)

            if "red" in sensor_name:
                red_humanoid_raycast_ids.extend(data_indices)

            if "blue" in sensor_name:
                blue_humanoid_raycast_ids.extend(data_indices)

        # Convert the lists to JAX arrays
        self._red_humanoid_qpos_ids = jp.array(red_humanoid_qpos_ids)
        self._red_humanoid_qvel_ids = jp.array(red_humanoid_qvel_ids)
        self._red_humanoid_body_ids = jp.array(red_humanoid_body_ids)
        self._red_humanoid_raycast_ids  = jp.array(red_humanoid_raycast_ids)
        self._red_humanoid_actuator_ids = jp.array(red_humanoid_actuator_ids)

        self._blue_humanoid_qpos_ids = jp.array(blue_humanoid_qpos_ids)
        self._blue_humanoid_qvel_ids = jp.array(blue_humanoid_qvel_ids)
        self._blue_humanoid_body_ids = jp.array(blue_humanoid_body_ids)
        self._blue_humanoid_raycast_ids  = jp.array(blue_humanoid_raycast_ids)
        self._blue_humanoid_actuator_ids = jp.array(blue_humanoid_actuator_ids)

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng1, rng2, rng3, rng4, rgn5 = jax.random.split(rng, 6)

        # Generate a radom global pose for both humanoids
        _, sampled_radius, sampled_theta = self.sample_positions_in_circle(rng1)
        x1 = sampled_radius[0] * jp.cos(sampled_theta[0])
        x2 = sampled_radius[1] * jp.cos(sampled_theta[1])
        y1 = sampled_radius[0] * jp.sin(sampled_theta[0])
        y2 = sampled_radius[1] * jp.sin(sampled_theta[1])
        z  = self._start_pose[2]

        azimuth = jax.random.uniform(rng2, (2,), minval=0, maxval=2 * jp.pi)
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
            "red_humanoid_last_action": jp.zeros(len(self._red_humanoid_actuator_ids)),
            "blue_humanoid_last_action": jp.zeros(len(self._blue_humanoid_actuator_ids)),
        }
        obs = self._calculate_obs(data, info)
        metrics = {"velocity": zero}
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, actions: jax.Array) -> mjx_env.State:
        data0  = state.data
        data   = mjx_env.step(self.mjx_model, data0, actions, self.n_substeps)

        # BUG: Not sure if a dict works here, i'll explore dtypes on the alg side, this is just a place holder for now
        red_humanoid_actions = actions["red_humanoid"]
        blue_humanoid_actions = actions["blue_humanoid"]

        state.info["red_humanoid_last_action"] = red_humanoid_actions
        state.info["blue_humanoid_last_action"] = blue_humanoid_actions

        obs    = self._calculate_obs(data, state.info)
        reward = self._calculate_reward(data) * self.dt
        termination = self._calculate_termination(data)
        return mjx_env.State(data, obs, reward, termination, state.metrics, state.info)

    def _calculate_termination(self, data: mjx.Data) -> jax.Array:
        # Terminate if one of the humanoids is out of the ring, for now use root position, in the future we'll use  foot contact position or something
        blue_humanoid_dist = jp.linalg.norm(data.qpos[self._blue_humanoid_qpos_ids[0:2]])
        red_humanoid_dist = jp.linalg.norm(data.qpos[self._red_humanoid_qpos_ids[0:2]])
        humanoid_out = jp.logical_or(blue_humanoid_dist > self._radius, red_humanoid_dist > self._radius)
        
        # Terminate if one of the bodies rotates to far
        blue_humanoid_torso_angle  = jp.ravel(data.site_xmat[self._blue_humanoid_site_index])[8]
        red_humanoid_torso_angle   = jp.ravel(data.site_xmat[self._red_humanoid_site_index])[8]

        blue_humanoid_fallen = blue_humanoid_torso_angle < self._max_tilt
        red_humanoid_fallen  = red_humanoid_torso_angle < self._max_tilt

        humanoid_fallen = jp.logical_or(blue_humanoid_fallen, red_humanoid_fallen)

        reset_condition = jp.logical_or(humanoid_fallen, humanoid_out)
        return jp.where(reset_condition, 1.0, 0.0)

    def _calculate_reward(self, data: mjx.Data) -> jax.Array:
        # Pass
        existence = 1.0
        return jp.array(existence)

    # TODO: critic and actor netowrk will need vastly diifferent observations
    # actor will have agent specific, obs, im thinking of sectioning this into two specific dicts, 
    # one for each agent, then the alg runner will index correctly, then a global critic state input, this will depend on the alg multi-agent style that i go with
    def _calculate_obs(
        self, data: mjx.Data, info: Dict[str, jax.Array]
    ) -> mjx_env.Observation:

        # Get the local base linear velocity of both humanoids
        red_humanoid_linvel  = mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_local_linvel")
        blue_humanoid_linvel = mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_local_linvel")

        # Get the local base angular velocity of both humanoids
        red_humanoid_angvel  = mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_local_angvel")
        blue_humanoid_angvel = mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_local_angvel")

        # Relative joint positions for both humanoid
        red_humanoid_joint_positions  = data.qpos[self._red_humanoid_qpos_ids[7:]] - self._start_pose[self._red_humanoid_qpos_ids[7:]]
        blue_humanoid_joint_positions = data.qpos[self._blue_humanoid_qpos_ids[7:]] - self._start_pose[self._blue_humanoid_qpos_ids[7:]]

        # Joint velocities for both humanoid
        red_humanoid_joint_velocities = data.qvel[self._red_humanoid_qvel_ids[6:]]
        blue_humanoid_joint_velocities = data.qvel[self._blue_humanoid_qvel_ids[6:]]

        # projected_gravity for each humanoid
        red_humanoid_gravity = data.site_xmat[self._red_humanoid_site_index].T @ jp.array([0, 0, -1])
        blue_humanoid_gravity = data.site_xmat[self._blue_humanoid_site_index].T @ jp.array([0, 0, -1])

        # Foot contact and force observations.
        blue_humanoid_foot_contact = jp.array([
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_left_foot_contact")[0] > 0,
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_right_foot_contact")[0] > 0,
        ])

        red_humanoid_foot_contact = jp.array([
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_left_foot_contact")[0] > 0,
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_right_foot_contact")[0] > 0,
        ])


        red_humanoid_foot_forces = jp.concatenate([
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_left_foot_force"),
            mjx_env.get_sensor_data(self._mj_model, data, "red_humanoid_right_foot_force"),
        ])

        blue_humanoid_foot_forces = jp.concatenate([
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_left_foot_force"),
            mjx_env.get_sensor_data(self._mj_model, data, "blue_humanoid_right_foot_force"),
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

        red_humanoid_obs = jp.concatenate([
            red_humanoid_linvel,
            red_humanoid_angvel,
            red_humanoid_gravity,
            red_humanoid_joint_positions,
            red_humanoid_joint_velocities,
            red_humanoid_foot_contact,
            red_humanoid_foot_forces,
            red_humanoid_last_action,
            red_humanoid_raycast_obs
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
            blue_humanoid_raycast_obs
        ])

        #TODO: Critic, wholle lotta info
        critic_obs = jp.array([21])

        return {
            "red": red_humanoid_obs,
            "blue": blue_humanoid_obs,
            "critic" : critic_obs
        }

    def sample_positions_in_circle(self, key, radius = 5, too_close = 0.5):
        # Sample two position within circle, resample is they are too close
        # x = rcos(theta)
        # y = rsin(theta)

        def sample(args):
            key, _, _ = args
            key,key1,key2 = jax.random.split(key,3)
            sampled_radius = jax.random.uniform(key1, (2,), minval=0, maxval=radius)
            sampled_theta = jax.random.uniform(key2, (2,), minval=-jp.pi, maxval=jp.pi)
            return key, sampled_radius, sampled_theta, 


        def collision(args):
            _, sampled_radius, sampled_theta = args
            x1 = sampled_radius[0] * jp.cos(sampled_theta[0])
            x2 = sampled_radius[1] * jp.cos(sampled_theta[1])
            y1 = sampled_radius[0] * jp.sin(sampled_theta[0])
            y2 = sampled_radius[1] * jp.sin(sampled_theta[1])

            dist = jp.sqrt((x2-x1) ** 2 + (y2-y1) ** 2)
            return dist < too_close

        first_sample = sample((key, jp.zeros(2), jp.zeros(2)))

        return jax.lax.while_loop(
            collision,
            sample,
            first_sample
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

locomotion.register_environment('multi_agent_sumo', Sumo, default_config)
