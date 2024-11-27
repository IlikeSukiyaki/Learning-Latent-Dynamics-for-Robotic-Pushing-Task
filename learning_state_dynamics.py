import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE
import math

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = []
    # --- Your code here

    for _ in tqdm(range(num_trajectories), desc="Collecting data"):
        # Reset the environment and get the initial state
        states = []
        actions = []
        state = env.reset()
        states.append(state)

        for _ in range(trajectory_length):
            # Sample a random action within the action space limits
            action = env.action_space.sample()

            # Apply the action to the environment and collect the next state
            next_state, _, done, _ = env.step(action)

            # Append the state and action
            actions.append(action)
            states.append(next_state)

            if done:
                break

        # Convert states and actions to numpy arrays and ensure they're float32
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)

        # Store this trajectory's data
        collected_data.append({
            'states': states,
            'actions': actions
        })

    # ---
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    # train_loader = None
    # val_loader = None
    # --- Your code here
    # Instantiate the dataset
    dataset = SingleStepDynamicsDataset(collected_data)

    # Split the dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, state_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, action_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multi-step data.
    :return:
    """
    # Instantiate the MultiStepDynamicsDataset
    dataset = MultiStepDynamicsDataset(collected_data, num_steps=num_steps)

    # Split the dataset into 80% training and 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for both datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        # Flatten all (state, action, next_state) pairs from all trajectories into a single list.
        self.samples = []
        for trajectory in collected_data:
            states = trajectory['states']
            actions = trajectory['actions']
            for t in range(len(actions)):
                self.samples.append({
                    'state': torch.tensor(states[t], dtype=torch.float32),
                    'action': torch.tensor(actions[t], dtype=torch.float32),
                    'next_state': torch.tensor(states[t + 1], dtype=torch.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multi-step trajectory torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the multi-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.num_steps = num_steps
        self.samples = []

        # Flatten all subtrajectories into a single list of samples
        for trajectory in self.data:
            states = trajectory['states']
            actions = trajectory['actions']
            trajectory_length = len(states) - num_steps  # The number of valid subtrajectories
            for t in range(trajectory_length):
                self.samples.append({
                    'state': torch.tensor(states[t], dtype=torch.float32),
                    'action': torch.tensor(actions[t:t + num_steps], dtype=torch.float32),
                    'next_state': torch.tensor(states[t + 1:t + 1 + num_steps], dtype=torch.float32),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}
    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.rg = math.sqrt((block_length ** 2 + block_width ** 2) / 12)

    def forward(self, pose_pred, pose_target):
        # pose_pred and pose_target are expected to have shape (batch_size, 3) where 3 represents [x, y, theta]
        mse_x = F.mse_loss(pose_pred[:, 0], pose_target[:, 0])
        mse_y = F.mse_loss(pose_pred[:, 1], pose_target[:, 1])
        mse_theta = F.mse_loss(pose_pred[:, 2], pose_target[:, 2])

        # Combine the MSE losses with the radius of gyration scaling for theta
        se2_pose_loss = mse_x + mse_y + self.rg * mse_theta
        return se2_pose_loss


class SingleStepLoss(nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn  # The loss function passed in, e.g., SE2PoseLoss

    def forward(self, model, state, action, target_state):
        """
        Compute the single-step loss resultant of querying the model with (state, action) and comparing the predictions with target_state.

        :param model: The dynamics model, e.g., AbsoluteDynamicsModel.
        :param state: The current state (batch of states).
        :param action: The action taken (batch of actions).
        :param target_state: The ground truth next state (batch of states).
        :return: The computed loss for this batch.
        """
        # --- Your code here
        # Predict the next state given the current state and action
        predicted_state = model(state, action)

        # Compute the loss between the predicted state and the target (ground truth) state
        single_step_loss = self.loss(predicted_state, target_state)
        # ---

        return single_step_loss


class MultiStepLoss(nn.Module):
    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn  # The loss function passed in, e.g., SE2PoseLoss
        self.discount = discount  # Discount factor for future steps (optional)

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.

        :param model: The dynamics model, e.g., AbsoluteDynamicsModel or ResidualDynamicsModel.
        :param state: The initial state (batch of states).
        :param actions: The sequence of actions (batch_size, num_steps, action_dim).
        :param target_states: The sequence of target states (batch_size, num_steps, state_dim).
        :return: The computed multi-step loss for this batch.
        """
        batch_size, num_steps, _ = actions.shape
        multi_step_loss = 0.0  # Initialize the total loss

        current_state = state  # Start from the initial state

        for step in range(num_steps):
            # Get the action for this time step
            current_action = actions[:, step, :]

            # Predict the next state using the current state and action
            predicted_next_state = model(current_state, current_action)

            # Get the target state for this time step
            target_next_state = target_states[:, step, :]

            # Compute the loss for this step
            step_loss = self.loss(predicted_next_state, target_next_state)

            # Apply the discount factor (optional)
            discounted_step_loss = (self.discount ** step) * step_loss

            # Accumulate the discounted loss
            multi_step_loss += discounted_step_loss

            # Update the current state to the predicted state for the next iteration
            current_state = predicted_next_state

        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # Neural network layers
        self.fc1 = nn.Linear(state_dim + action_dim, 100)  # Input layer combining state and action
        self.fc2 = nn.Linear(100, 100)  # Hidden layer
        self.fc3 = nn.Linear(100, state_dim)  # Output layer predicting the next state

    def forward(self, state, action):
        # Concatenate state and action as input
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))  # Apply ReLU to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU to the second layer
        x = self.fc3(x)  # Final layer with no activation
        return x


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics: x_{t+1} = x_{t} + f(x_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define the layers
        self.fc1 = nn.Linear(state_dim + action_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, state_dim)

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to the provided state.
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Pass through the network layers with ReLU activations
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Predict the state difference
        state_difference = self.fc3(x)

        # Add the state difference to the current state to get the next state
        next_state = state + state_difference

        return next_state


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    Q = torch.tensor([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.1]])  # The diagonal weight matrix

    # Calculate the difference between the current state and the target state
    error = state - target_pose

    # Compute the quadratic cost: error^T * Q * error
    cost = torch.einsum('bi,ij,bj->b', error, Q, error)
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w

    # Extract the (x, y, theta) from the state
    x, y, theta = state[:, 0], state[:, 1], state[:, 2]

    # Compute the corners of the pushed block (rotated rectangle)
    half_box_size = box_size / 2

    # Corners of the block before rotation
    block_corners = torch.tensor([
        [-half_box_size, -half_box_size],
        [half_box_size, -half_box_size],
        [half_box_size, half_box_size],
        [-half_box_size, half_box_size]
    ]).T  # Shape: (2, 4)

    # Rotation matrix based on theta
    cos_theta = torch.cos(theta).view(-1, 1, 1)
    sin_theta = torch.sin(theta).view(-1, 1, 1)
    rotation_matrix = torch.cat([
        torch.cat([cos_theta, -sin_theta], dim=-1),
        torch.cat([sin_theta, cos_theta], dim=-1)
    ], dim=-2)  # Shape: (B, 2, 2)

    # Rotate the block corners
    rotated_corners = torch.matmul(rotation_matrix, block_corners)  # Shape: (B, 2, 4)
    block_corners_rotated = rotated_corners + state[:, :2].unsqueeze(-1)  # Add the (x, y) translation

    # Compute the bounding box of the block
    min_block = block_corners_rotated.min(dim=-1)[0]  # Shape: (B, 2)
    max_block = block_corners_rotated.max(dim=-1)[0]  # Shape: (B, 2)

    # Compute the bounding box of the obstacle
    min_obs = obstacle_centre - obstacle_dims / 2  # Shape: (2,)
    max_obs = obstacle_centre + obstacle_dims / 2  # Shape: (2,)

    # Check for overlap between the block and the obstacle (axis-aligned bounding box)
    overlap_x = (min_block[:, 0] < max_obs[0]) & (max_block[:, 0] > min_obs[0])
    overlap_y = (min_block[:, 1] < max_obs[1]) & (max_block[:, 1] > min_obs[1])

    # If both x and y overlaps are true, then there is a collision
    in_collision = overlap_x & overlap_y  # Shape: (B,)
    in_collision = in_collision.float()  # Convert to float for consistency

    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]], dtype=torch.float32)  # Weighting matrix Q
    in_collision_penalty = 100  # Penalty for being in collision

    # Compute the quadratic cost (xt − xgoal)^T Q (xt − xgoal)
    error = state - target_pose
    quadratic_cost = torch.sum(error @ Q * error, dim=1)

    # Compute the collision penalty
    in_collision = collision_detection(state)
    collision_cost = in_collision_penalty * in_collision

    # Sum the quadratic cost and the collision cost
    cost = quadratic_cost + collision_cost

    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        # Predict the next state using the learned model
        next_state = self.model(state, action)
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be sent to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        # Convert the input state to a torch tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Shape (1, state_size)

        # Query the MPPI controller for the optimal action
        action_tensor = self.mppi.command(state_tensor)

        # Convert the action tensor back to a numpy array and remove the batch dimension
        action = action_tensor.squeeze(0).detach().numpy()
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here



# ---
# ============================================================
