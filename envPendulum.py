import numpy as np

class PendulumEnv:
    """
    Inverted Pendulum Environment

    The pendulum starts in a random position and the goal is to apply torque
    to swing it into an upright position with its center of gravity above the fixed point.

    State: [cos(theta), sin(theta), theta_dot]
    Action: torque in range [-max_torque, max_torque]
    """

    def __init__(self, g=10.0, max_torque=2.0, dt=0.05, max_speed=8.0, m=1.0, l=1.0):
        """
        Initialize the pendulum environment.

        Args:
            g: gravity acceleration (m/s^2)
            max_torque: maximum torque that can be applied
            dt: timestep for integration
            max_speed: maximum angular velocity
            m: mass of the pendulum
            l: length of the pendulum
        """
        self.g = g
        self.max_torque = max_torque
        self.dt = dt
        self.max_speed = max_speed
        self.m = m
        self.l = l

        # State variables
        self.theta = None  # angle (radians)
        self.theta_dot = None  # angular velocity (radians/s)
        self.steps = 0
        self.max_steps = 200

    def reset(self, seed=None, theta_range=(-np.pi, np.pi)):
        """
        Reset the environment to initial state.

        Args:
            seed: random seed for reproducibility
            theta_range: range for initial angle (low, high)

        Returns:
            observation: [cos(theta), sin(theta), theta_dot]
        """
        if seed is not None:
            np.random.seed(seed)

        # Random initial angle and velocity
        self.theta = np.random.uniform(low=theta_range[0], high=theta_range[1])
        self.theta_dot = np.random.uniform(low=-1.0, high=1.0)
        self.steps = 0

        return self._get_obs()

    def _get_obs(self):
        """
        Get current observation.

        Returns:
            observation: [cos(theta), sin(theta), theta_dot]
        """
        return np.array([np.cos(self.theta), np.sin(self.theta), self.theta_dot], dtype=np.float32)

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: torque to apply (will be clipped to [-max_torque, max_torque])

        Returns:
            observation: next state [cos(theta), sin(theta), theta_dot]
            reward: reward for this step
            done: whether episode is finished
            info: additional information dictionary
        """
        # Clip action to valid range
        torque = np.clip(action, -self.max_torque, self.max_torque)

        # Store current state for reward calculation
        theta = self.theta
        theta_dot = self.theta_dot

        # Physics simulation using Euler integration
        # Equation of motion: theta_ddot = (3*g / (2*l)) * sin(theta) + (3 / (m*l^2)) * torque
        theta_ddot = (3 * self.g / (2 * self.l)) * np.sin(theta) + (3.0 / (self.m * self.l ** 2)) * torque

        # Update angular velocity and clip to max speed
        self.theta_dot = theta_dot + theta_ddot * self.dt
        self.theta_dot = np.clip(self.theta_dot, -self.max_speed, self.max_speed)

        # Update angle
        self.theta = theta + self.theta_dot * self.dt

        # Normalize angle to [-pi, pi]
        self.theta = self._angle_normalize(self.theta)

        # Calculate reward
        # Goal: minimize angle from vertical (0), angular velocity, and torque usage
        reward = -(self.theta ** 2 + 0.1 * self.theta_dot ** 2 + 0.001 * (torque ** 2))

        # Update step counter
        self.steps += 1
        done = self.steps >= self.max_steps

        # Additional info
        info = {
            'theta': self.theta,
            'theta_dot': self.theta_dot,
            'torque': torque
        }

        return self._get_obs(), reward, done, info

    @staticmethod
    def _angle_normalize(angle):
        """
        Normalize angle to [-pi, pi].

        Args:
            angle: angle in radians

        Returns:
            normalized angle in range [-pi, pi]
        """
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def get_state_description(self):
        """
        Get human-readable description of current state.

        Returns:
            dict with state information
        """
        theta_deg = np.degrees(self.theta)
        return {
            'angle_rad': self.theta,
            'angle_deg': theta_deg,
            'angular_velocity': self.theta_dot,
            'steps': self.steps,
            'position_description': self._describe_position()
        }

    def _describe_position(self):
        """Describe the pendulum position in human-readable form."""
        angle_deg = np.degrees(self.theta)
        if -10 <= angle_deg <= 10:
            return "upright (near goal)"
        elif -45 <= angle_deg <= 45:
            return "tilted upward"
        elif 135 <= angle_deg or angle_deg <= -135:
            return "hanging down"
        elif 45 < angle_deg <= 135:
            return "tilted right"
        else:
            return "tilted left"

class Environment:
    """
    MDP Environment Template.

    Students should implement:
    - step(): state transition logic
    - reward(): reward function
    """

    def __init__(self):
        # State space bounds
        self.S_low_limits = np.array([0., 0.])
        self.S_upper_limits = np.array([1., 1.])

        # Action space bounds
        self.A_low_limits = np.array([0.])
        self.A_upper_limits = np.array([1.])

        # Current state
        self.s = None

        # Episode settings
        self.max_steps = 200
        self.current_step = 0

        # Internal pendulum
        self._pendulum = PendulumEnv()

    def reset(self):
        """Reset environment to initial state. Returns initial state."""
        self.current_step = 0
        self._pendulum.reset()
        self.s = self._normalize_state()
        return self.s.copy()

    def _normalize_state(self):
        """Map pendulum state to [0,1]²."""
        s0 = (self._pendulum.theta + np.pi) / (2 * np.pi)
        s1 = (self._pendulum.theta_dot + self._pendulum.max_speed) / (2 * self._pendulum.max_speed)
        return np.array([np.clip(s0, 0, 1), np.clip(s1, 0, 1)])

    def _denormalize_action(self, action):
        """Map action from [0,1] to [-max_torque, max_torque]."""
        a = action[0] if isinstance(action, np.ndarray) else action
        return a * 2 * self._pendulum.max_torque - self._pendulum.max_torque

    def step(self, action):
        """
        Execute action and transition to next state.

        Args:
            action: Action in [A_low_limits, A_upper_limits]

        Returns:
            next_state: The resulting state
            reward: The reward for this transition
            done: Whether the episode has ended
            info: Additional information (dict)

        *** STUDENTS IMPLEMENT THIS ***
        """
        self.current_step += 1

        # Convert normalized action to torque
        torque = self._denormalize_action(action)

        # Step pendulum
        _, r, done, info = self._pendulum.step(torque)

        # Get normalized next state
        next_state = self._normalize_state()

        # Update state
        self.s = next_state

        return next_state.copy(), r, done, info
