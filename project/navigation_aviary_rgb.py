import os
import numpy as np
import pybullet as p
import pkg_resources

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary  import ActionType, ObservationType
from gym_pybullet_drones.envs.single_agent_rl.SingleVisionAviary import SingleVisionAviary

TABLE_HEIGHT = 0.29
TABLE_THICK  = 0.05

class NavigationAviaryVision(SingleVisionAviary):
    """Single agent RL problem: fly across an abyss"""
    
    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 goal_loc = [1, 5, 1],
                 num_obstacles = 0):
        """Initialization of a single agent RL environment.
        Using the generic single agent RL superclass.
        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)
        """
        # Set the goal location
        self.goal_location = goal_loc
        self.start_location = [0, 0, TABLE_HEIGHT + TABLE_THICK]
        # self.max_distance   = np.array(self.goal_location) - np.array(self.start_location)

        # # Init the base class
        initial_xyzs = np.zeros((1, 3))
        initial_xyzs[0] = np.array(self.start_location)

        # Define the number of obstacles
        self.num_obstacles = num_obstacles
        self.obstacles     = []
        self.crash         = False

        # Precalculate where to put those obstacles :)
        if self.num_obstacles > 0:
            # Get the heading vector 
            self.heading_vec = np.array(self.goal_location) + [0, 0, TABLE_HEIGHT + TABLE_THICK] - np.array(self.start_location)

            # Determine the depth of each obstacle, make it equally spaced
            self.obstacle_x = np.linspace(self.start_location[0] + self.heading_vec[0]/self.num_obstacles, self.heading_vec[0] - self.heading_vec[0]/self.num_obstacles, self.num_obstacles)
            self.obstacle_y = np.linspace(self.start_location[1] + self.heading_vec[1]/self.num_obstacles, self.heading_vec[1] - self.heading_vec[1]/self.num_obstacles, self.num_obstacles)
            self.obstacle_z = np.linspace(self.start_location[2] + self.heading_vec[2]/self.num_obstacles, self.heading_vec[2] - self.heading_vec[2]/self.num_obstacles, self.num_obstacles)

        # Parameters
        self.k1 = 1
        self.k2 = 10
        self.k3 = 10
        self.k4= 1
        self.k5 = -0.1
        self.k6 = 1

        super().__init__(drone_model=drone_model,
            initial_xyzs=initial_xyzs,
            initial_rpys=initial_rpys,
            physics=physics,
            freq=freq,
            aggregate_phy_steps=aggregate_phy_steps,
            gui=gui,
            record=record,
            obstacles=True,
            user_debug_gui=False)

    def _addPlatforms(self):
        # Add the platform that the drone sits on
        p.loadURDF("table/table.urdf",
                    [0, 0, 0],
                    p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT,
                    globalScaling = 0.5
                    )

        # Add the landing platform
        p.loadURDF("table/table.urdf",
                        np.array(self.goal_location) - np.array([0, 0, TABLE_HEIGHT + TABLE_THICK]),
                        p.getQuaternionFromEuler([0, 0, 0]),
                        physicsClientId=self.CLIENT,
                        globalScaling = 0.5
                        )

    ################################################################################
    
    def _addObstacles(self):
        """Add obstacles to the environment.
        Extends the superclass method and add the gate build of cubes and an architrave.
        """
        super()._addObstacles()
        self.crash = False
        self._addPlatforms()
        self.obstacles = []

        for i in range(self.num_obstacles):
            noise_offset = np.random.normal(0.0, 0.1)

            if i % 2 == 0:
                # Add obstacle in the middle
                obstacle = p.loadURDF("gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf",
                            [self.obstacle_x[i], self.obstacle_y[i], self.obstacle_z[i]],
                            p.getQuaternionFromEuler([0, 0, 0]),
                            physicsClientId=self.CLIENT
                            )
            else:
                # Add obstacles to the right of goal relative to the drone
                obstacle = p.loadURDF("gym-pybullet-drones/gym_pybullet_drones/assets/cf2x.urdf",
                            [self.obstacle_x[i], self.obstacle_y[i] + noise_offset, self.obstacle_z[i]],
                            p.getQuaternionFromEuler([0, 0, 0]),
                            physicsClientId=self.CLIENT
                            )
            self.obstacles.append(obstacle)


    def _physics(self, rpm, nth_drone):
        for obstacle in self.obstacles:   
            for i in range(4):
                p.applyExternalForce(obstacle,
                                    i,
                                    forceObj=[0, 0, .06615],
                                    posObj=[0, 0, 0],
                                    flags=p.LINK_FRAME,
                                    physicsClientId=self.CLIENT
                                    )

        return super()._physics(rpm, nth_drone)
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.
        Returns
        -------
        float
            The reward.
        """

        state = self._getDroneStateVector(0)
        norm_ep_time = (self.step_counter/self.SIM_FREQ) / self.EPISODE_LEN_SEC

        # Reward for the hover height
        R_hover = self.k1 * (np.exp(-(state[2] - self.goal_location[2]) ** 2))

        # Reward for goal
        goal_dist = np.linalg.norm(self.goal_location[0:2]-state[0:2])
        if goal_dist > 1:
            goal_reward = np.exp(-goal_dist**2)
        else:
            goal_reward = np.exp(-goal_dist ** 2)

        R_goal = self.k2 *  goal_reward # Quadraticly increase the reward when close, quadratically decrease when ar away

        # Give negative rewards for too much change in torque
        R_crash = self.k3 * -10 * int(state[2] <= 0.1 or state[2] >= 2.5)
        if R_crash < 0:
            self.crash = True

        # Give lesser reward/more penalthy depending on how much the orientation is changing
        # Right now, penalize -1 for a change of pi/4
        angular_vel = state[13:16]
        R_rot = -self.k4 * np.linalg.norm(angular_vel)

        # Reduce the wandering of the drone to the waypoint
        R_wander = self.k5

        # Calculate the drone direction and the direction to the goal
        pitch = state[8]
        yaw   = state[9]
        #drone_dir = np.array([np.sin(yaw) * np.cos(pitch), np.cos(yaw)*np.cos(pitch), np.sin(pitch)])
        velocity = state[10:13]
        goal_dir  = self.goal_location - state[0:3]     

        # Create into unit vectors then find the angle between these vectors for the reward
        heading_error = np.arccos(np.dot(velocity, goal_dir) / (np.linalg.norm(velocity) * np.linalg.norm(goal_dir)))

        theta_thresh = np.pi/8
        if np.abs(heading_error) < theta_thresh:
            heading_reward = min(1/(2*heading_error), 2)
        else:
            heading_reward = -2 * heading_error
        R_heading = self.k6 * heading_reward 

        #R_goal = 
        return R_hover + R_goal + R_crash + R_rot + R_wander + R_heading

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value.
        Returns
        -------
        bool
            Whether the current episode is done.
        """
        if self.crash or self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).
        Unused.
        Returns
        -------
        dict[str, int]
            Dummy value.
        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years

    ################################################################################
    
    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.
        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.
        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.
        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,   # XY Position
                                      normalized_pos_z,    # Z Position
                                      state[3:7],          # Quaternion repr. of pose of drone
                                      normalized_rp,       # Roll pitch
                                      normalized_y,        # Yaw
                                      normalized_vel_xy,   # First derivative of position for x and y
                                      normalized_vel_z,    # First derivative of altitude change
                                      normalized_ang_vel,  # How much the orientation of the drone is changing in WORLD FRAME
                                      state[16:20]         # Speed of the Motor (RPM)
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.
        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FlyThruGateAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))