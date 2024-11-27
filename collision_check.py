from learning_state_dynamics import collision_detection
import torch
from panda_pushing_env import PandaPushingEnv, TARGET_POSE_OBSTACLES, BOX_SIZE
from learning_state_dynamics import ResidualDynamicsModel, SE2PoseLoss, SingleStepLoss
import matplotlib.pyplot as plt
# let's test the collision detection with two states
# The first should be in collision, the second should not

states = torch.tensor([
    [0.6, 0.2, 0.0],
    [0.3, 0.15, 0.7]
])

collision = collision_detection(states)

print(f'First state in collision? {collision[0]}')
print(f'Second state in collision? {collision[1]}')

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from learning_state_dynamics import (
    PushingController,
    free_pushing_cost_function,
    obstacle_avoidance_pushing_cost_function,
    collision_detection
)


# Define the ImageVisualizer class
class ImageVisualizer(object):
    def __init__(self, save_folder="frames"):
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.frame_count = 0

    def set_data(self, img):
        # Save each frame as an image
        img_path = os.path.join(self.save_folder, f"frame_{self.frame_count:04d}.png")
        plt.imsave(img_path, img)
        self.frame_count += 1

    def reset(self):
        # Clear saved frames
        self.frame_count = 0
        for f in os.listdir(self.save_folder):
            os.remove(os.path.join(self.save_folder, f))

# Load the pre-trained model
state_dim = 3  # Assuming state_dim is 3 based on your problem setup
action_dim = 3  # Assuming action_dim is 3 based on your problem setup

# Instantiate the ResidualDynamicsModel
pushing_residual_dynamics_model = ResidualDynamicsModel(state_dim, action_dim)
# Specify the folder where frames will be stored
save_folder = "/home/yifeng/PycharmProjects/Learning_Dynamics/learning_dynamics_1/pushing_obstacle_frames"

# Load the pre-trained weights
model_path = "/home/yifeng/PycharmProjects/Learning_Dynamics/learning_dynamics_1/pushing_multi_step_residual_dynamics_model.pt"
pushing_residual_dynamics_model.load_state_dict(torch.load(model_path, weights_only=True))
pushing_residual_dynamics_model.eval()  # Set the model to evaluation mode

# Instantiate the ImageVisualizer
visualizer = ImageVisualizer(save_folder=save_folder)

# Initialize the environment and the controller
env = PandaPushingEnv(
    visualizer=visualizer,
    render_non_push_motions=False,
    include_obstacle=True,
    camera_heigh=800,
    camera_width=800,
    render_every_n_steps=5
)

controller = PushingController(
    env,
    pushing_residual_dynamics_model,
    obstacle_avoidance_pushing_cost_function,
    num_samples=1000,
    horizon=20
)

env.reset()

# Reset the environment and get the initial state
state_0 = env.reset()
state = state_0

# Maximum number of steps
num_steps_max = 25

# Run the control loop
for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

# Evaluate if the goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_OBSTACLES
goal_distance = np.linalg.norm(end_state[:2] - target_state[:2])  # Evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')

# Close the figure
# plt.close(fig)

