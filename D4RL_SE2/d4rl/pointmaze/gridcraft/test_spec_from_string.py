from d4rl.pointmaze.gridcraft import grid_spec
# from d4rl.pointmaze.gridcraft import grid_spec_se2
import numpy as np

def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = grid_spec.WALL
            elif tile == 'G':
                maze_arr[w][h] = grid_spec.GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = grid_spec.EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr


if __name__ == "__main__":
    U_MAZE = \
            "#####\\"+\
            "#GOO#\\"+\
            "###O#\\"+\
            "#OOO#\\"+\
            "#####"
    gs = grid_spec.spec_from_string(U_MAZE)
    # print(gs)
    # print gs by iterating over width and height
    for i in range(gs.width):
        for j in range(gs.height):
            print(gs[i,j], end=',')
        print()
    maze_arr = parse_maze(U_MAZE)
    empty_locations = list(zip(*np.where(maze_arr == grid_spec.EMPTY)))
    
    empty_locations.sort()

    # print empty_locations
    print(empty_locations)

    idx = np.random.choice(len(empty_locations))
    # reset_location = np.array(self.empty_locations[idx]).astype(self.observation_space.dtype)
    reset_location = np.array(empty_locations[idx])
    reset_location_center = reset_location + np.array([0.5,0.5])
    print(reset_location)
    target_location = reset_location_center + np.random.uniform(low=-.1, high=.1, size=len(reset_location_center))
    print(target_location)


