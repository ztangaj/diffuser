import gym

O_MAZE = \
        "########\\"+\
        "#OOOOOO#\\"+\
        "#OOOOOO#\\"+\
        "#OO##OO#\\"+\
        "#OO##OO#\\"+\
        "#OOOOOO#\\"+\
        "#OOOOOO#\\"+\
        "########"

ENVIRONMENT_SPECS = (
    {
        'id': 'HopperFullObs-v2',
        'entry_point': ('diffuser.environments.hopper:HopperFullObsEnv'),
    },
    {
        'id': 'HalfCheetahFullObs-v2',
        'entry_point': ('diffuser.environments.half_cheetah:HalfCheetahFullObsEnv'),
    },
    {
        'id': 'Walker2dFullObs-v2',
        'entry_point': ('diffuser.environments.walker2d:Walker2dFullObsEnv'),
    },
    {
        'id': 'AntFullObs-v2',
        'entry_point': ('diffuser.environments.ant:AntFullObsEnv'),
    },
    {
        'id': 'maze2d-se2-omaze-v0',
        'entry_point': ('diffuser.environments.grid_env_se2:GridEnvSE2'),
        'max_episode_steps': 150, 
        'kwargs': {
            'maze_str':O_MAZE,
            'dataset_path': '/homes/ztangaj/tony/replandiffuser/diffuser/D4RL_SE2/maze2d-se2-omaze-v0.hdf5',
        }
    }, 
    {
        'id': 'maze2d-se2-omaze-interpolate',
        'entry_point': ('diffuser.environments.grid_env_se2:GridEnvSE2'),
        'max_episode_steps': 150, 
        'kwargs': {
            'maze_str':O_MAZE,
            'dataset_path': '/homes/ztangaj/tony/replandiffuser/diffuser/maze2d-se2-omaze-interpolate.hdf5',
        }
    },
)

def register_environments():
    try:
        for environment in ENVIRONMENT_SPECS:
            gym.register(**environment)

        gym_ids = tuple(
            environment_spec['id']
            for environment_spec in  ENVIRONMENT_SPECS)

        return gym_ids
    except:
        print('[ diffuser/environments/registration ] WARNING: not registering diffuser environments')
        return tuple()