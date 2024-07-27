import typing as tp
from pathlib import Path

import gymnasium as gym
import numpy as np


class SimpleMazeEnv(gym.Env):
    def __init__(self, seed: int = 13):
        super(SimpleMazeEnv, self).__init__()
        self.maze_path: Path = Path('').resolve() / 'assets' / 'map.txt'
        assert self.maze_path.exists(), 'The path to the maze file is incorrect. Please make sure you are in the base directory as the project.'
        self.maze: np.ndarray = np.loadtxt(str(self.maze_path)).astype(int)

        self.seed: int = seed

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(4)
        self.observation_space: gym.spaces.Tuple = gym.spaces.Tuple(spaces=[gym.spaces.Discrete(n=self.maze.size), gym.spaces.Discrete(n=2)],
                                                                    seed=self.seed)

        self.action_to_dir: dict[int, tuple[int, int]] = {0: (1, 0), 1: (0, 1), 2: (0, -1), 3: (-1, 0)}

        self.curr_player_coord: int | None = None

    @property
    def width(self):
        return self.maze.shape[0]

    @property
    def height(self):
        return self.maze.shape[1]

    def mapping_1d_to_2d(self, coord: int) -> tuple[int, int]:
        x: int = coord // self.height
        y: int = coord % self.height
        return x, y

    def mapping_2d_to_1d(self, xy: tuple[int, int]) -> int:
        coord: int = xy[0] * self.height + xy[1]
        return coord

    def is_start(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        if coord is not None:
            return self.mapping_1d_to_2d(coord) == (0, 0)
        elif x is not None and y is not None:
            return x == 0 and y == 0
        else:
            raise ValueError('Either coords or x and y must be specified.')

    def get_cell(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> int:
        if coord is not None:
            x, y = self.mapping_1d_to_2d(coord)
        elif x is None and y is None:
            raise ValueError('Either coords or x and y must be specified.')
        return self.maze[x, y]

    def is_wall(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        cell_val: int = self.get_cell(coord=coord, x=x, y=y)
        return cell_val == 1

    def is_empty(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        return not self.is_wall(coord=coord, x=x, y=y)

    def is_goal(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        return self.get_cell(coord=coord, x=x, y=y) == 3

    def get_observation(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> tuple[int, int]:
        if coord is None and x is not None and y is not None:
            coord: int = self.mapping_2d_to_1d((x, y))
        elif coord is None:
            raise ValueError('Either coords or x and y must be specified.')
        return coord, int(self.is_goal(coord=coord))

    def is_visitable_location(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        if coord is not None:
            x, y = self.mapping_1d_to_2d(coord)
        return 0 <= x < self.width and 0 <= y < self.height and self.is_empty(x=x, y=y)

    def get_new_location(self, action: int, coord: int | None = None) -> tuple[int, int]:
        if coord is None:
            coord = self.curr_player_coord
        x, y = self.mapping_1d_to_2d(coord)
        dx, dy = self.action_to_dir[action]
        return x + dx, y + dy

    def step(self, action: int):
        new_x, new_y = self.get_new_location(action=action)
        if self.is_visitable_location(x=new_x, y=new_y):
            self.curr_player_coord = self.mapping_2d_to_1d((new_x, new_y))
        obs: tuple[int, int] = self.get_observation(coord=self.curr_player_coord)
        reward: int = -1 if not self.is_goal(coord=self.curr_player_coord) else 10
        terminated: bool = self.is_goal(coord=self.curr_player_coord)
        return obs, reward, terminated, False, {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super(SimpleMazeEnv, self).reset(seed=seed or self.seed, options=options)
        self.curr_player_coord = self.mapping_2d_to_1d((0, 0))
        return self.get_observation(coord=self.curr_player_coord), {}

    def render(self, mode="human"):
        pass


class MazeEnv(gym.Env):
    def __init__(self, seed: int = 13):
        super(MazeEnv, self).__init__()
        self.maze_path: Path = Path('').resolve() / 'assets' / 'map.txt'
        assert self.maze_path.exists(), 'The path to the maze file is incorrect. Please make sure you are in the base directory as the project.'
        self.maze: np.ndarray = np.loadtxt(str(self.maze_path)).astype(int)

        self.seed: int = seed

        self.action_space: gym.spaces.Discrete = gym.spaces.Discrete(4)
        self.observation_space: gym.spaces.Tuple = gym.spaces.Tuple(spaces=[gym.spaces.Discrete(n=self.width), gym.spaces.Discrete(n=self.height)],
                                                                    seed=self.seed)

        self.action_to_dir: dict[int, tuple[int, int]] = {0: (1, 0), 1: (0, 1), 2: (0, -1), 3: (-1, 0)}

        self.curr_player_coord: int | None = None

    @property
    def width(self):
        return self.maze.shape[0]

    @property
    def height(self):
        return self.maze.shape[1]

    def mapping_1d_to_2d(self, coord: int) -> tuple[int, int]:
        x: int = coord // self.height
        y: int = coord % self.height
        return x, y

    def mapping_2d_to_1d(self, xy: tuple[int, int]) -> int:
        coord: int = xy[0] * self.height + xy[1]
        return coord

    def is_start(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        if coord is not None:
            return self.mapping_1d_to_2d(coord) == (0, 0)
        elif x is not None and y is not None:
            return x == 0 and y == 0
        else:
            raise ValueError('Either coords or x and y must be specified.')

    def get_cell(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> int:
        if coord is not None:
            x, y = self.mapping_1d_to_2d(coord)
        elif x is None and y is None:
            raise ValueError('Either coords or x and y must be specified.')
        return self.maze[x, y]

    def is_wall(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        cell_val: int = self.get_cell(coord=coord, x=x, y=y)
        return cell_val == 1

    def is_empty(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        return not self.is_wall(coord=coord, x=x, y=y)

    def is_goal(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        return self.get_cell(coord=coord, x=x, y=y) == 3

    def get_observation(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> tuple[int, int]:
        if coord is None and x is not None and y is not None:
            coord: int = self.mapping_2d_to_1d((x, y))
        elif coord is None:
            raise ValueError('Either coords or x and y must be specified.')
        return self.mapping_1d_to_2d(coord)

    def is_visitable_location(self, *, coord: int | None = None, x: int | None = None, y: int | None = None) -> bool:
        if coord is not None:
            x, y = self.mapping_1d_to_2d(coord)
        return 0 <= x < self.width and 0 <= y < self.height and self.is_empty(x=x, y=y)

    def get_new_location(self, action: int, coord: int | None = None) -> tuple[int, int]:
        if coord is None:
            coord = self.curr_player_coord
        x, y = self.mapping_1d_to_2d(coord)
        dx, dy = self.action_to_dir[action]
        return x + dx, y + dy

    def step(self, action: int):
        new_x, new_y = self.get_new_location(action=action)
        if self.is_visitable_location(x=new_x, y=new_y):
            self.curr_player_coord = self.mapping_2d_to_1d((new_x, new_y))
        obs: tuple[int, int] = self.get_observation(coord=self.curr_player_coord)
        reward: int = -1 if not self.is_goal(coord=self.curr_player_coord) else 10
        terminated: bool = self.is_goal(coord=self.curr_player_coord)
        return obs, reward, terminated, False, {}

    def reset(self, seed: int | None = None, options: dict | None = None):
        super(MazeEnv, self).reset(seed=seed or self.seed, options=options)
        self.curr_player_coord = self.mapping_2d_to_1d((0, 0))
        return self.get_observation(coord=self.curr_player_coord), {}

    def render(self, mode="human"):
        pass
