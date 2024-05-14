import sys
import numpy as np
import gym
import gym.spaces
import pygame

from D4RL_SE2.d4rl.pointmaze.gridcraft import grid_spec_se2
from D4RL_SE2.d4rl.pointmaze.gridcraft.grid_spec_se2 import EMPTY, WALL, TILES, START, RENDER_DICT, REWARD

ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4
ACT_CW = 5
ACT_CCW = 6
ACT_DICT = {
    ACT_NOOP: [0, 0, 0],
    ACT_UP: [0, -1, 0],
    ACT_LEFT: [-1, 0, 0],
    ACT_RIGHT: [+1, 0, 0],
    ACT_DOWN: [0, +1, 0],
    ACT_CW: [0, 0, +1], # 1 means rotating for THETA_GRID_RES rad
    ACT_CCW: [0, 0, -1]
}
ACT_TO_STR = {
    ACT_NOOP: 'NOOP',
    ACT_UP: 'UP',
    ACT_LEFT: 'LEFT',
    ACT_RIGHT: 'RIGHT',
    ACT_DOWN: 'DOWN',
    ACT_CW: 'ROT_CW',
    ACT_CCW: 'ROT_CCW'
}

SEGMENT_LENGTH = 1.5

class TransitionModel(object):
    def __init__(self, gridspec, eps=0.2):
        self.gs = gridspec
        self.eps = eps

    def get_aprobs(self, s, a):
        legal_moves = self.__get_legal_moves(s)
        p = np.zeros(len(ACT_DICT))
        p[list(legal_moves)] = self.eps / (len(legal_moves))
        if a in legal_moves:
            p[a] += 1.0-self.eps
        else:
            #p = np.array([1.0,0,0,0,0])  # NOOP
            p[ACT_NOOP] += 1.0-self.eps
        return p

    def __get_legal_moves(self, s):
        xytheta = np.array(self.gs.idx_to_xytheta(s))
        moves = set()
        for move in ACT_DICT:
            if ACT_DICT[move] == ACT_NOOP:
                moves.add(move)
                continue

            new_s = xytheta + ACT_DICT[move]

            # Check if the two end points of the segment located at new_s is legal: within the grid and not a wall
            segment_center = new_s[0:2] + np.array([0.5,0.5])
            segment_theta = new_s[2] * grid_spec_se2.THETA_GRID_RES
            segment_end1 = segment_center + 0.5*SEGMENT_LENGTH*np.array([np.cos(segment_theta), np.sin(segment_theta)])
            segment_end2 = segment_center - 0.5*SEGMENT_LENGTH*np.array([np.cos(segment_theta), np.sin(segment_theta)])
            end1_grid = (int(np.floor(segment_end1[0])), int(np.floor(segment_end1[1])))
            end2_grid = (int(np.floor(segment_end2[0])), int(np.floor(segment_end2[1])))

            if not self.gs.out_of_bounds(new_s[0:2]) and self.gs[new_s[0:2]] != WALL and not self.gs.out_of_bounds(end1_grid) and not self.gs.out_of_bounds(end2_grid) and self.gs[end1_grid] != WALL and self.gs[end2_grid] != WALL:
                moves.add(move)

        moves.add(ACT_NOOP)
        return moves


class RewardFunction(object):
    def __init__(self, rew_map=None, default=0):
        if rew_map is None:
            rew_map = {
                REWARD: 1.0
            }
        self.default = default
        self.rew_map = rew_map

    def __call__(self, gridspec, s, a, ns):
        xy = gridspec.idx_to_xytheta(s)[0:2]
        x,y = xy
        val = gridspec[x,y]
        if val in self.rew_map:
            return self.rew_map[val]
        return self.default


class GridEnvSE2(gym.Env):
    def __init__(self, maze_str,
                 tiles=TILES,
                 rew_fn=None,
                 teps=0.0, 
                 max_timesteps=None,
                 rew_map=None,
                 terminal_states=None,
                 default_rew=0):
        self._env_args = {'teps': teps, 'max_timesteps': max_timesteps}
        self.str_maze_spec = maze_str
        print('maze_str:', maze_str)
        #type
        print('type(maze_str):', type(maze_str))
        self.gs = grid_spec_se2.spec_from_string(maze_str)
        # TODO : DONE
        self.num_states = len(self.gs) * grid_spec_se2.THETA_GRID_SIZE
        self.num_actions = 7
        self.model = TransitionModel(self.gs, eps=teps)
        self.terminal_states = terminal_states
        if rew_fn is None:
            rew_fn = RewardFunction(rew_map=rew_map, default=default_rew)
        self.rew_fn = rew_fn
        self.possible_tiles = tiles
        self.max_timesteps = max_timesteps
        self._timestep = 0
        self._true_q = None  # q_vals for debugging
        self._target = np.array([0,0])
        self._state_xytheta = np.array([0,0,0])
        self.qpos = np.array([0.0,0.0,0.0])
        self.qvel = np.array([0.0,0.0,0.0])
        self.init_qvel = np.array([0.0,0.0,0.0])
        self.maze_arr = grid_spec_se2.parse_maze(maze_str)
        self.empty_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.empty_locations.sort()
        self.wall_locations = list(zip(*np.where(self.maze_arr == WALL)))
        self.wall_locations.sort()
        self.collision_check_sample = 5

        self.window_size = 512
        self.window = None
        self.clock = None
        self.render_mode = "human"
        self.render_fps = 30
        self.size = np.maximum(self.gs.width, self.gs.height)
        super(GridEnvSE2, self).__init__()


    def step_stateless(self, s, a, verbose=False):
        aprobs = self.model.get_aprobs(s, a)
        samp_a = np.random.choice(range(self.num_actions), p=aprobs)
        print('samp_a:', samp_a)
        print('ACT_DICT[samp_a]:', ACT_DICT[samp_a])

        next_s = self.gs.idx_to_xytheta(s) + ACT_DICT[samp_a]
        if next_s[2] > 2*np.pi:
            next_s[2] -= 2*np.pi
        if next_s[2] < 0:
            next_s[2] += 2*np.pi
        
        print('next_s:', next_s)

        next_s_xy_noisy = next_s[0:2] + np.array([0.5,0.5]) + self.np_random.uniform(low=-.1, high=.1, size=len(next_s[0:2]))
        # (waypoint[2] + 0.5) * grid_spec_se2.THETA_GRID_RES
        next_s_theta_noisy = (next_s[2] + 0.5) * grid_spec_se2.THETA_GRID_RES + self.np_random.uniform(low=-.1, high=.1)
        qpos = np.concatenate((next_s_xy_noisy, [next_s_theta_noisy]))
        qvel = self.qvel
        self.set_state(qpos, qvel, next_s)

        next_s_idx = self.gs.xytheta_to_idx(next_s)
        rew = self.rew_fn(self.gs, s, samp_a, next_s_idx)

        if verbose:
            print('Act: %s. Act Executed: %s' % (ACT_TO_STR[a], ACT_TO_STR[samp_a]))
        return next_s_idx, next_s, rew

    # TODO: DONE
    def step(self, a, verbose=False):
        ns, next_s, r = self.step_stateless(self.__state, a, verbose=verbose)
        traj_infos = {}
        obs = self._get_obs()

        done = False
        self._timestep += 1
        if self.max_timesteps is not None:
            if self._timestep >= self.max_timesteps:
                done = True
        return obs, r, done, traj_infos

    #  TODO: DONE
    # def reset(self):
    #     start_idxs = np.array(np.where(self.gs.spec == START)).T
    #     start_idx = start_idxs[np.random.randint(0, start_idxs.shape[0])]
    #     start_idx = self.gs.xy_to_idx(start_idx)
    #     self.__state =start_idx
    #     self._timestep = 0
    #     return start_idx #flat_to_one_hot(start_idx, len(self.gs))
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._timestep = 0

        ob = self.reset_model()
        return ob

    def reset_model(self):
        feasible = False
        qpos = np.array([0.0,0.0,0.0])
        qvel = np.array([0.0,0.0,0.0])
        count = 0

        while not feasible:
            # sample position from empty grids
            idx = self.np_random.choice(len(self.empty_locations))
            reset_location = np.array(self.empty_locations[idx])
            # initialize an np.array with values 2,1
            # reset_location = np.array([2,1])
            reset_location_center_noisy = reset_location + np.array([0.5,0.5]) + self.np_random.uniform(low=-.1, high=.1, size=len(reset_location))

            # sample feasible theta, try for 10 times
            for _ in range(10):
                # sample theta from 0 to 2pi
                theta = (_+0.5) * np.pi / 5
                theta_idx = int(np.floor(theta / grid_spec_se2.THETA_GRID_RES))
                if not self.collision_check_segment(reset_location_center_noisy, theta):
                    qpos = np.concatenate((reset_location_center_noisy, np.array([theta])))
                    feasible = True
                    break
            count += 1
            if count > 100:
                raise ValueError('Cannot find feasible initial state')
            
        qvel = self.init_qvel + self.np_random.randn(len(self.init_qvel)) * .1
        xytheta = np.concatenate((reset_location, [theta_idx]))
        self.set_state(qpos, qvel, xytheta)
        # if self.reset_target:
        #     self.set_target()
        return self._get_obs()
    

    def collision_check_segment(self, pos, theta):
        for i in range(self.collision_check_sample):
            step = (i+1) * 0.5 * SEGMENT_LENGTH / float(self.collision_check_sample)
            pt_check1 = pos + np.array([step*np.cos(theta), step*np.sin(theta)])
            pt_check2 = pos - np.array([step*np.cos(theta), step*np.sin(theta)])
            if self.collision_check_point(pt_check1) or self.collision_check_point(pt_check2):
                return True
        return False


    def collision_check_point(self, pos):
        int_array = np.floor(pos).astype(int)
        if self.gs[int_array] == WALL:
            return True
        else: 
            return False


    def set_state(self, qpos, qvel, state_xytheta):
        self.qpos = qpos
        self.qvel = qvel
        self._state_xytheta = state_xytheta
        self.__state = self.gs.xytheta_to_idx(state_xytheta)

    def _get_obs(self):
        return np.concatenate([self.qpos, self.qvel]).ravel()
    
    # TODO : DONE

    def render(self, start, target, waypoints):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw target
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (target + 0.5) * pix_square_size,
            pix_square_size / 8,
        )

        # Draw Wall
        for wall in self.wall_locations:
            pygame.draw.rect(
                canvas,
                (128, 128, 128),
                pygame.Rect(
                    pix_square_size * np.array(wall),
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw start state
        rect_center = (pix_square_size * (start[0] + 0.5), pix_square_size * (start[1] + 0.5))
        rotation_angle = start[2] * grid_spec_se2.THETA_GRID_RES
        seg_len_pixel = SEGMENT_LENGTH * pix_square_size
        endpt1 = rect_center + np.array([0.5*seg_len_pixel*np.cos(rotation_angle), 0.5*seg_len_pixel*np.sin(rotation_angle)])
        endpt2 = rect_center - np.array([0.5*seg_len_pixel*np.cos(rotation_angle), 0.5*seg_len_pixel*np.sin(rotation_angle)])
        pygame.draw.line(canvas, (0, 255, 0), endpt1, endpt2, width=6)

        # Draw agent waypoints
        rect_width, rect_height = pix_square_size * SEGMENT_LENGTH, pix_square_size * 0.1
        for waypoint in waypoints:
            rect_center = (pix_square_size * (waypoint[0] + 0.5), pix_square_size * (waypoint[1] + 0.5))
            rotation_angle = waypoint[2] * grid_spec_se2.THETA_GRID_RES
            seg_len_pixel = SEGMENT_LENGTH * pix_square_size
            endpt1 = rect_center + np.array([0.5*seg_len_pixel*np.cos(rotation_angle), 0.5*seg_len_pixel*np.sin(rotation_angle)])
            endpt2 = rect_center - np.array([0.5*seg_len_pixel*np.cos(rotation_angle), 0.5*seg_len_pixel*np.sin(rotation_angle)])
            pygame.draw.line(canvas, (0, 0, 255), endpt1, endpt2, width=6)

        # Draw gridlines
        for x in range(self.gs.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (pix_square_size * self.gs.width, pix_square_size * x),
                width=3,
            )
        for x in range(self.gs.width + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, pix_square_size * self.gs.height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # save the image
            pygame.image.save(canvas, "test3.png")

            # Wait for a key press
            key_pressed = False
            while not key_pressed:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        key_pressed = True

            self.close()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            # self.clock.tick(self.render_fps)

        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    # TODO: DONE
    def set_target(self, target_location=None):
        if target_location is None:
            feasible_target = True;

            for _ in range(100):
                feasible_target = True
                idx = self.np_random.choice(len(self.empty_locations))
                # reset_location = np.array(self.empty_locations[idx]).astype(self.observation_space.dtype)
                reset_location = np.array(self.empty_locations[idx])
                # reset_location_center = reset_location + np.array([0.5,0.5])
                # target_location = reset_location_center + self.np_random.uniform(low=-.1, high=.1, size=len(reset_location_center))
                target_location = reset_location + np.array([0.5,0.5])

                print('target_location:', target_location)

                # check sampled feasible theta
                for __ in range(10):
                    # sample theta from 0 to 2pi
                    theta = (__+0.5) * np.pi / 5
                    theta_idx = int(np.floor(theta / grid_spec_se2.THETA_GRID_RES))
                    if self.collision_check_segment(target_location, theta):
                        print('collision_check_segment for theta', theta)
                        feasible_target = False
                        break
                
                if feasible_target:
                    print('feasible_target:', target_location)
                    break
            
            if not feasible_target:
                raise ValueError('Cannot find feasible target state')
            
        self._target = target_location

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)

    @property
    def observation_space(self):
        dO = len(self.gs)
        #return gym.spaces.Box(0,1,shape=dO)
        return gym.spaces.Discrete(dO)

    # TODO: DONE
    def get_transitions(self, s, a):
        aprobs = self.model.get_aprobs(s, a)
        t_dict = {}
        for sa in range(self.num_actions):
            if aprobs[sa] > 0:
                next_s = self.gs.idx_to_xytheta(s) + ACT_DICT[sa]
                # round rotation to [0, 2pi]
                if next_s[2] > 2*np.pi:
                    next_s[2] -= 2*np.pi
                if next_s[2] < 0:
                    next_s[2] += 2*np.pi
                next_s_idx = self.gs.xytheta_to_idx(next_s)
                t_dict[next_s_idx] = t_dict.get(next_s_idx, 0.0) + aprobs[sa]
        return t_dict
    
    def transition_matrix(self):
        """Constructs this environment's transition matrix.

        Returns:
          A dS x dA x dS array where the entry transition_matrix[s, a, ns]
          corrsponds to the probability of transitioning into state ns after taking
          action a from state s.
        """
        ds = self.num_states
        da = self.num_actions
        transition_matrix = np.zeros((ds, da, ds))
        for s in range(ds):
            for a in range(da):
                transitions = self.get_transitions(s,a)
                for next_s in transitions:
                    transition_matrix[s, a, next_s] = transitions[next_s]
        return transition_matrix

    def reward_matrix(self):
        """Constructs this environment's reward matrix.

        Returns:
          A dS x dA x dS numpy array where the entry reward_matrix[s, a, ns]
          reward given to an agent when transitioning into state ns after taking
          action s from state s.
        """
        ds = self.num_states
        da = self.num_actions
        rew_matrix = np.zeros((ds, da, ds))
        for s in range(ds):
            for a in range(da):
                for ns in range(ds):
                    rew_matrix[s, a, ns] = self.rew_fn(self.gs, s, a, ns)
        return rew_matrix