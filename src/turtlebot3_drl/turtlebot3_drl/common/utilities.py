from queue import Empty
from turtlebot3_msgs.srv import DrlStep
from turtlebot3_msgs.srv import Goal
from std_srvs.srv import Empty
import os
import time
import rclpy
import torch
import numpy
from transformers import AutoTokenizer, AutoModel
from ..common.settings import REWARD_FUNCTION, COLLISION_OBSTACLE, COLLISION_WALL, TUMBLE, SUCCESS, TIMEOUT, RESULTS_NUM
import yaml
import random


def select_goal_and_sentence(data):
    """
    Selects a random goal and corresponding sentence pair from the YAML data.

    Args:
        data (dict): Parsed YAML data as a dictionary.

    Returns:
        dict: A dictionary with the selected section, goal, and sentence.
    """
    # Randomly select a section from the data
    section = random.choice(list(data.keys()))
    section_data = data[section]

    # Extract the goal and sentences for the selected section
    goal = section_data.get('goal', {})
    sentences = section_data.get('sentences', [])

    # Randomly select a sentence from the sentences list
    sentence = random.choice(sentences) if sentences else "No sentence available"

    return {
        "section": section,
        "goal": goal,
        "sentence": sentence
    }


def get_goal_and_sentence():
    with open(os.getenv('DRLNAV_BASE_PATH') + '/warehouse_mapping.yml') as stream:
        try:
            # print(yaml.safe_load(stream))
            data = yaml.safe_load(stream)
            return select_goal_and_sentence(data)
        except yaml.YAMLError as e:
            print(e)


import xml.etree.ElementTree as ET

try:
    with open('/tmp/drlnav_current_stage.txt', 'r') as f:
        stage = int(f.read())
except FileNotFoundError:
    print("\033[1m" + "\033[93m" + "Make sure to launch the gazebo simulation node first!" + "\033[0m}")

# Load a pre-trained language model
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def encode_text(text_prompt):
    inputs = tokenizer(text_prompt, return_tensors="pt")
    outputs = model(**inputs)
    # Use mean pooling to get a single vector
    embeddings = outputs.last_hidden_state.mean(dim=1)
    # Convert to a Python list of floats
    return embeddings.squeeze().tolist()

def check_gpu():
    print("gpu torch available: ", torch.cuda.is_available())
    if (torch.cuda.is_available()):
        print("device name: ", torch.cuda.get_device_name(0))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def step(agent_self, action, previous_action):
    req = DrlStep.Request()
    req.action = action
    req.previous_action = previous_action

    while not agent_self.step_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('env step service not available, waiting again...')
    future = agent_self.step_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                # print(res)
                return res.state, res.reward, res.done, res.success, res.distance_traveled
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting step service response!")

def init_episode(agent_self):
    state, _, _, _, _ = step(agent_self, [], [0.0, 0.0])
    return state

def get_goal_status(agent_self):
    req = Goal.Request()
    while not agent_self.goal_comm_client.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('new goal service not available, waiting again...')
    future = agent_self.goal_comm_client.call_async(req)

    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            if future.result() is not None:
                res = future.result()
                return res.new_goal
            else:
                agent_self.get_logger().error(
                    'Exception while calling service: {0}'.format(future.exception()))
                print("ERROR getting   service response!")

def wait_new_goal(agent_self):
    while(get_goal_status(agent_self) == False):
        print("Waiting for new goal... (if persists: reset gazebo_goals node)")
        time.sleep(1.0)

def pause_simulation(agent_self, real_robot):
    if real_robot:
        return
    while not agent_self.gazebo_pause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('pause gazebo service not available, waiting again...')
    future = agent_self.gazebo_pause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def unpause_simulation(agent_self, real_robot):
    if real_robot:
        return
    while not agent_self.gazebo_unpause.wait_for_service(timeout_sec=1.0):
        agent_self.get_logger().info('unpause gazebo service not available, waiting again...')
    future = agent_self.gazebo_unpause.call_async(Empty.Request())
    while rclpy.ok():
        rclpy.spin_once(agent_self)
        if future.done():
            return

def translate_outcome(outcome):
    if outcome == SUCCESS:
        return "SUCCESS"
    elif outcome == COLLISION_WALL:
        return "COLL_WALL"
    elif outcome == COLLISION_OBSTACLE:
        return "COLL_OBST"
    elif outcome == TIMEOUT:
        return "TIMEOUT"
    elif outcome == TUMBLE:
        return "TUMBLE"
    else:
        return f"UNKNOWN: {outcome}"

# --- Environment ---

def euler_from_quaternion(quat):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quat = [x, y, z, w]
    """
    x = quat.x
    y = quat.y
    z = quat.z
    w = quat.w

    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = numpy.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w*y - z*x)
    if sinp < -1:
        sinp = -1
    if sinp > 1:
        sinp = 1
    pitch = numpy.arcsin(sinp)

    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = numpy.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

def get_scan_count():
    tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_burger/model.sdf')

    root = tree.getroot()
    for link in root.find('model').findall('link'):
        if link.get('name') == 'base_scan':
            return int(link.find('sensor').find('ray').find('scan').find('horizontal').find('samples').text)

def get_simulation_speed(stage):
    tree = ET.parse(os.getenv('DRLNAV_BASE_PATH') + '/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_drl_stage' + str(stage) + '/burger.model')
    root = tree.getroot()
    real_time_factor = int(root.find('world').find('physics').find('real_time_factor').text)
    print(f'real_time_factor: {real_time_factor}')
    return real_time_factor