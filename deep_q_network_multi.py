#!/usr/bin/env python
from __future__ import print_function

# import tensorflow as tf
import tensorflow.compat.v1 as tf
import os
import json
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
import multiprocessing
from collections import deque

tf.disable_v2_behavior()
GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = 1
RANDOM_ACTION_INTERVAL = 10
RANDOM_ACTION_PROB = 0.1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def sample_biased_offset(min_offset=20, max_offset=400, k=50, s=0.0005):
    offsets = np.arange(min_offset, max_offset + 1)
    weights = np.zeros_like(offsets, dtype=np.float32)

    for i, x in enumerate(offsets):
        if x <= k:
            weights[i] = 1.0  # flat region
        else:
            val = 1.0 - s * (x - k)
            weights[i] = max(val, 0.0)  # clip to 0
    
    weights /= weights.sum()  # Normalize
    return int(np.random.choice(offsets, p=weights))

def trainNetwork(s, readout, h_fc1, sess, process_id=0, base_episode_offset=0):
    # define the cost function
    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    game_state = game.GameState()
    base_save_dir = f"frame_action_data/episodes/round11_proc{process_id}"
    r = 11
    os.makedirs(base_save_dir, exist_ok=True)
    # frame_action_pairs = []  # Store (frame, action) pairs
    image2 = cv2.imread('./assets/sprites/resized_color_background.jpg')
    D = deque()
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")
    # Setup for recording full-resolution .mp4 videos
    video_writer = None
    video_segment = 0
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_size = None  # Will be determined dynamically
    recording_episode = False
    
    
    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    t0 = sample_biased_offset()
    use_random_mode = False
    
    # USE_RANDOM_PATTERN_PROB = 0.05
    USE_RANDOM_PATTERN_PROB = 0.50
    PATTERN_UP_MEAN = 1  #2
    PATTERN_NONE_MEAN = 20  #5
    random_pattern_counter = 0
    current_pattern_action = 0
    pattern_switch_frame = 0
    while "flappy bird" != "angry bird":
        # choose an action epsilon greedily
        if t > OBSERVE:
            break
        if t > t0:
            use_random_mode = True
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if use_random_mode and random_pattern_counter > 0:
            a_t[current_pattern_action] = 1
            action_index = current_pattern_action
            random_pattern_counter -= 1
            if random_pattern_counter == pattern_switch_frame:
                current_pattern_action = 0
        elif t % FRAME_PER_ACTION == 0:
            if use_random_mode and random.random() < USE_RANDOM_PATTERN_PROB:
                n = max(1, int(np.random.normal(PATTERN_UP_MEAN, 1)))
                m = max(1, int(np.random.normal(PATTERN_NONE_MEAN, 2)))
                random_pattern_counter = n + m
                pattern_switch_frame = m
                current_pattern_action = 1
                a_t[1] = 1
                action_index = 1
                print(f"---- Pure Random Pattern ---- Flap {n} frames, then None {m} frames")
        
            elif random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange(ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing

        # scale down epsilon
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        # run the selected action and observe next state and reward
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        
        # Preprocess and rotate
        full_frame = cv2.rotate(x_t1_colored, cv2.ROTATE_90_CLOCKWISE)
        full_frame = cv2.cvtColor(full_frame, cv2.COLOR_RGB2BGR)
        
        if frame_size is None:
            frame_size = (full_frame.shape[1], full_frame.shape[0])
        
        
        # Start recording if needed
        if not recording_episode:
            x = t0-t
            y = random.randint(int(0.1*x), int(0.2*x))
            z = t
            truncating_index = t + y
            episode_dir = os.path.join(base_save_dir, f"episode{video_segment+1+base_episode_offset}_r{r}")
            os.makedirs(episode_dir, exist_ok=True)
            video_filename = os.path.join(episode_dir, f"video_{(video_segment+1+base_episode_offset):04d}_r{r}.mp4")
            video_writer = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            frame_action_pairs = []
            recording_episode = True
            print(f"[INFO] Started recording: {video_filename}")
        
        
        # Write frame to video
        if recording_episode and r_t != -1:
            mask = cv2.inRange(full_frame, (0, 0, 0), (0, 0, 0))
            result = full_frame.copy()
            result[mask > 0] = image2[mask > 0]
            video_writer.write(result)

            frame_filename = os.path.join(episode_dir, f"frame_{t}_r{r}.png")
            cv2.imwrite(frame_filename, result)
            frame_action_pairs.append({
                "frame": frame_filename,
                "action": int(action_index)
            })
        
        # Handle crash
        if r_t == -1:
            print("Detected failure frame. Generating synthetic failure scene...")
            print("[INFO] Episode ended. Saving video...")
            failure_frame = cv2.GaussianBlur(result, (21, 21), 0)
            text = "GAME OVER"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            font_thickness = 4
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (failure_frame.shape[1] - text_size[0]) // 2
            text_y = (failure_frame.shape[0] + text_size[1]) // 2
            cv2.putText(failure_frame, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)

            # 3. Save and pad the failure frame multiple times
            failure_pattern_counter = 0
            failure_pattern_switch_frame = 0
            failure_pattern_action = 0
            num_pad_frames = random.randint(32, 48)
            for i in range(num_pad_frames):
                padded_frame_filename = os.path.join(episode_dir, f"frame_{t}_failpad_{i}_r{r}.png")
                cv2.imwrite(padded_frame_filename, failure_frame)
                video_writer.write(failure_frame)
                # === Use same randomized movement pattern ===
                if failure_pattern_counter > 0:
                    action = failure_pattern_action
                    failure_pattern_counter -= 1
                    if failure_pattern_counter == failure_pattern_switch_frame:
                        failure_pattern_action = 0  # Switch to 'do nothing'
                else:
                    # Generate a new movement pattern: flap n, none m
                    n = max(1, int(np.random.normal(PATTERN_UP_MEAN, 1)))
                    m = max(1, int(np.random.normal(PATTERN_NONE_MEAN, 2)))
                    failure_pattern_counter = n + m
                    failure_pattern_switch_frame = m
                    failure_pattern_action = 1  # Start with flap
                    action = 1
                    print(f"[FAILPAD] New Pattern: Flap {n} frames, then None {m} frames")
                
                frame_action_pairs.append({
                    "frame": padded_frame_filename,
                    "action": int(action)  # Using -1 to indicate terminal frame
                })
                
            # Save metadata
            meta = {
                "random_agent_index": int(x + z + 1),
                "num_pad_frames": int(num_pad_frames),
                "total_frames": len(frame_action_pairs),
            }
            
            with open(os.path.join(episode_dir, "meta.json"), "w") as f:
                json.dump(meta, f, indent=4)
            
            with open(os.path.join(episode_dir, "actions.json"), "w") as f:
                json.dump(frame_action_pairs, f, indent=4)
            video_writer.release()
            video_writer = None
            video_segment += 1
            recording_episode = False

            # 5. Prepare for next episode
            t0 = t + sample_biased_offset()
            use_random_mode = False
            random_pattern_counter = 0
            current_pattern_action = 0
            pattern_switch_frame = 0
                
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
        
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # update the old values
        s_t = s_t1
        t += 1
        
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-dqn', global_step = t)
           

        # print info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files

def run_worker(process_id, base_episode_offset):
    print(f"[Process {process_id}] Starting...")
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    trainNetwork(s, readout, h_fc1, sess, process_id, base_episode_offset)


def playGameParallel(num_processes=4):
    processes = []
    for i in range(num_processes):
        base_episode_offset = i * 1000  # Ensure no filename conflicts
        p = multiprocessing.Process(target=run_worker, args=(i, base_episode_offset))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    playGameParallel(num_processes=4)
