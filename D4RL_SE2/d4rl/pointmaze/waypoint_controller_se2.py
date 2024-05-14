import numpy as np
from d4rl.pointmaze import q_iteration
from D4RL_SE2.d4rl.pointmaze.gridcraft import grid_env_se2
from D4RL_SE2.d4rl.pointmaze.gridcraft import grid_spec_se2


ZEROS = np.zeros((2,), dtype=np.float32)
ONES = np.zeros((2,), dtype=np.float32)


class WaypointController(object):
    def __init__(self, maze_str, solve_thresh=0.3, p_gain=10.0, d_gain=-1.0):
        self.maze_str = maze_str
        self._target = -1000 * ONES

        self.p_gain = p_gain
        self.d_gain = d_gain
        self.solve_thresh = solve_thresh
        self.vel_thresh = 0.1

        self._waypoint_idx = 0
        self._waypoints = []
        self._waypoint_prev_loc = ZEROS

        self.env = grid_env_se2.GridEnvSE2(maze_str)

    def current_waypoint(self):
        return self._waypoints[self._waypoint_idx]

    # TODO:
    def get_action(self, start_state, velocity, target):
        if np.linalg.norm(self._target - target) > 1e-1: 
            print('New target!', target, 'old:', self._target)
            self._new_target(start_state, target)

        dist = np.linalg.norm(start_state[0:2] - self._target[0:2])
        print('dist:', dist)
        print('self.solve_thresh:', self.solve_thresh)
        vel = self._waypoint_prev_loc[0:2] - start_state[0:2]
        vel_norm = np.linalg.norm(vel)
        task_not_solved = (dist >= self.solve_thresh) # or (vel_norm >= self.vel_thresh)
        print('task_not_solved:', task_not_solved)

        if task_not_solved:
            print('self._waypoint_idx:', self._waypoint_idx)
            next_wpnt = self._waypoints[self._waypoint_idx]
        else:
            next_wpnt = np.concatenate((self._target, [start_state[2]]))

        next_wpnt_idx = self.gridify_state_xytheta(next_wpnt)
        # Compute control
        # prop = next_wpnt - start_state
        prop = np.array(next_wpnt_idx) - np.array(self.gridify_state_xytheta(start_state))
        print('prop:', prop)
        print('next_wpnt:', next_wpnt)
        print('start_state:', start_state)
        print('len(self._waypoints): ', len(self._waypoints))

        # action = self.p_gain * prop + self.d_gain * velocity
        action = prop
        print('action:', action)

        dist_next_wpnt = np.linalg.norm(start_state[0:2] - next_wpnt[0:2])
        # if task_not_solved and (dist_next_wpnt < self.solve_thresh) and (vel_norm<self.vel_thresh):
        if task_not_solved: #and (dist_next_wpnt < self.solve_thresh) and (vel_norm<self.vel_thresh):
            print('self._waypoint_idx+++++++++++++++++++++++++++++++++')
            self._waypoint_idx += 1
            if self._waypoint_idx == len(self._waypoints)-1:
                assert np.linalg.norm(self._waypoints[self._waypoint_idx][0:2] - self._target) <= self.solve_thresh

        self._waypoint_prev_loc = start_state
        action = np.clip(action, -1.0, 1.0)
        print('action:', action)

        # round action to int
        # print('action:', action)

        return action, (not task_not_solved)

    def gridify_state_xy(self, state):
        # return (int(round(state[0])), int(round(state[1])))
        return (int(np.floor(state[0])), int(np.floor(state[1])))
    
    def gridify_state_xytheta(self, state):
        return (int(np.floor(state[0])), int(np.floor(state[1])), int(np.floor(state[2]/grid_spec_se2.THETA_GRID_RES)))

    def _new_target(self, start_state, target):
        print('Computing waypoints from %s to %s' % (start_state, target))
        start_grid = self.gridify_state_xytheta(start_state)
        print('start_grid:', start_grid)
        start_xytheta_idx = self.env.gs.xytheta_to_idx(start_grid)
        print('start_xytheta_idx:', start_xytheta_idx)
        target_grid = self.gridify_state_xy(target)
        print('target_grid:', target_grid)
        target_xy_idx = self.env.gs.xy_to_idx(target_grid)
        print('target_xy_idx:', target_xy_idx)
        self._waypoint_idx = 0

        self.env.gs[target_grid] = grid_spec_se2.REWARD
        q_values = q_iteration.q_iteration(env=self.env, num_itrs=50, discount=0.99)
        # load from file
        # q_values = np.load('/home/eason/workspace/d4rl_ws/D4RL/d4rl/pointmaze/q_values.npy')
        # save to file
        # np.save('/home/eason/workspace/d4rl_ws/D4RL/d4rl/pointmaze/q_values.npy', q_values)

        # compute waypoints by performing a rollout in the grid
        max_ts = 100
        s = start_xytheta_idx
        waypoints = []
        for i in range(max_ts):
            a = np.argmax(q_values[s])
            print('s:', s)
            print('a:', a)
            new_s, new_s_xytheta, reward = self.env.step_stateless(s, a)
            print('new_s:', new_s)

            waypoint = new_s_xytheta
            print('waypoint:', waypoint)

            # waypoint = self.env.gs.idx_to_xytheta(new_s)
            waypoint_xy_index = self.env.gs.xy_to_idx(waypoint[0:2])
            # if waypoint_xy_index != target_xy_idx:
            #     #  TODO: what?
            #     # waypoint = waypoint - np.random.uniform(size=(3,))*0.2
            #     waypoint = waypoint + np.random.uniform(low=-.1, high=.1, size=len(waypoint))+ np.array([0.5,0.5,0.])
            waypoint_x = waypoint[0] + 0.5
            waypoint_y = waypoint[1] + 0.5
            waypoint_theta = (waypoint[2] + 0.5) * grid_spec_se2.THETA_GRID_RES
            waypoints.append(np.array((waypoint_x, waypoint_y, waypoint_theta), dtype=np.float32))
            s = new_s
            if waypoint_xy_index == target_xy_idx:
                break
        self.env.gs[target_grid] = grid_spec_se2.EMPTY
        self._waypoints = waypoints
        self._waypoint_prev_loc = start_state
        self._target = target


        print('waypoints: ', waypoints)

        print('END of _new_target')

        # render waypoints
        # self.env.render(start_state, target, waypoints)

        



if __name__ == "__main__":
    print(q_iteration.__file__)
    TEST_MAZE = \
            "######\\"+\
            "#OOOO#\\"+\
            "#O##O#\\"+\
            "#OOOO#\\"+\
            "######"
    TEST_MAZE2 = \
            "########\\"+\
            "#OOOOOO#\\"+\
            "#OOOOOO#\\"+\
            "#OO##OO#\\"+\
            "#OO##OO#\\"+\
            "#OOOOOO#\\"+\
            "#OOOOOO#\\"+\
            "########"
    
    # TEST_MAZE
    # controller = WaypointController(TEST_MAZE)
    # start = np.array((2,1,0), dtype=np.float32)
    # target = np.array((4,3), dtype=np.float32)

    # TEST_MAZE2
    controller = WaypointController(TEST_MAZE2)
    # Test 1
    # start = np.array((2,1,0), dtype=np.float32)
    # target = np.array((5,5), dtype=np.float32)
    # target = np.array((4,5), dtype=np.float32)
    # Test 2
    # start = np.array((5,2,2), dtype=np.float32)
    # target = np.array((2,5), dtype=np.float32)

    # [1.5042861  4.50641951 1.25663706] to [4 2]
    # start = np.array((1.5042861, 4.50641951, 1.25663706), dtype=np.float32)
    # target = np.array((4,2), dtype=np.float32)

    # Computing waypoints from [2.50135406 2.57407144 0.31415927] to [2.5 1.5]
    # start = np.array((2.50135406, 2.57407144, 0.31415927), dtype=np.float32)
    # target = np.array((2.5, 1.5), dtype=np.float32)

    # Computing waypoints from [1.43009161 4.47586018 1.57079633] to [3.5 2.5]
    start = np.array((1.48299343, 3.56697621, 1.9644112), dtype=np.float32)
    target = np.array((5.5, 2.5), dtype=np.float32)

    velocity = np.array((0,0,0), dtype=np.float32)
    act, done = controller.get_action(start, velocity, target)
    # act, done = controller.get_action(start, target)
    print('act:', act)
    print('wpt:', controller._waypoints)
    print(act, done)

    # import pdb; pdb.set_trace()
    # pass

