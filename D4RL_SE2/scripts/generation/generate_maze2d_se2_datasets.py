import gym
import logging
# from d4rl.pointmaze import waypoint_controller
# from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse
from D4RL_SE2.d4rl.pointmaze import waypoint_controller_se2
from D4RL_SE2.d4rl.pointmaze.gridcraft import grid_env_se2
from D4RL_SE2.d4rl.pointmaze.gridcraft.grid_env_se2 import ACT_DICT


O_MAZE = \
        "########\\"+\
        "#OOOOOO#\\"+\
        "#OOOOOO#\\"+\
        "#OO##OO#\\"+\
        "#OO##OO#\\"+\
        "#OOOOOO#\\"+\
        "#OOOOOO#\\"+\
        "########"

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            }

def append_data(data, s, a, tgt, done, qpos, qvel):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(qpos.ravel().copy())
    data['infos/qvel'].append(qvel.ravel().copy())

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-se2-omaze-v0', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    #  TODO: DONE
    # env = gym.make(args.env_name)
    # maze = env.str_maze_spec
    # max_episode_steps = env._max_episode_steps
    maze = O_MAZE
    max_episode_steps = 100

    controller = waypoint_controller_se2.WaypointController(maze)
    env = grid_env_se2.GridEnvSE2(maze)

    env.set_target()
    s = env.reset()
    # act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    for _ in range(args.num_samples):
        print('sample count:', _)
        position = s[0:3]
        velocity = s[3:6]
        print('position:', position)
        print('velocity:', velocity)
        act, done = controller.get_action(position, velocity, env._target)
        print('act type:', type(act))
        print('act:', act)
        # print('wpt:', controller._waypoints)

        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.5

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True
        append_data(data, s, act, env._target, done, env.qpos, env.qvel)

        print('s:', s)
        print('act type:', type(act))

        # find act in ACT_DICT
        for act_idx, act_val in ACT_DICT.items():
            if np.all(act == act_val):
                break
        
        print('act:', act)
        print('act_idx:', act_idx)

        ns, _, _, _ = env.step(act_idx)

        # if len(data['observations']) % 10000 == 0:
        print(len(data['observations']))

        ts += 1
        if done:
            print('donefor one sample')
            env.set_target()
            done = False
            ts = 0
        else:
            s = ns

        # if args.render:
            # env.render()

    
    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name
    else:
        fname = '%s.hdf5' % args.env_name
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')


if __name__ == "__main__":
    main()
