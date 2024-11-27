import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from learning_state_dynamics import (
    PushingController, 
    free_pushing_cost_function, 
    collision_detection, 
    obstacle_avoidance_pushing_cost_function
)
from learning_state_dynamics import ResidualDynamicsModel, SE2PoseLoss, SingleStepLoss
from panda_pushing_env import PandaPushingEnv, TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE

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

# Specify the folder where frames will be stored
save_folder = "/home/yifeng/PycharmProjects/Learning_Dynamics/pushing_frames"

# Instantiate the ImageVisualizer
visualizer = ImageVisualizer(save_folder=save_folder)

# Load the pre-trained model
state_dim = 3  # Assuming state_dim is 3 based on your problem setup
action_dim = 3  # Assuming action_dim is 3 based on your problem setup

# Instantiate the ResidualDynamicsModel
pushing_residual_dynamics_model = ResidualDynamicsModel(state_dim, action_dim)

# Load the pre-trained weights
model_path = "/home/yifeng/PycharmProjects/Learning_Dynamics/learning_dynamics_1/pushing_residual_dynamics_model.pt"
pushing_residual_dynamics_model.load_state_dict(torch.load(model_path, weights_only=True))
pushing_residual_dynamics_model.eval()  # Set the model to evaluation mode

# Initialize the environment and the controller
env = PandaPushingEnv(
    visualizer=visualizer, 
    render_non_push_motions=False,  
    camera_heigh=800, 
    camera_width=800, 
    render_every_n_steps=5
)
controller = PushingController(env, pushing_residual_dynamics_model, free_pushing_cost_function, num_samples=100, horizon=10)
env.reset()

# Reset the environment and get the initial state
state_0 = env.reset()
state = state_0

# Maximum number of steps
num_steps_max = 30

# Run the control loop
for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    if done:
        break

# Evaluate if the goal is reached
end_state = env.get_state()
target_state = TARGET_POSE_FREE
goal_distance = np.linalg.norm(end_state[:2] - target_state[:2])  # Evaluate only position, not orientation
goal_reached = goal_distance < BOX_SIZE

print(f'GOAL REACHED: {goal_reached}')
