from typing import Tuple, Optional
import numpy as np


class HPPModel:

    # Векторы скоростей
    DIRECTIONS = np.array([
        [0, 1],   # 0: вправо
        [-1, 0],  # 1: вверх
        [0, -1],  # 2: влево
        [1, 0]    # 3: вниз
    ], dtype=np.int32)

    def __init__(self, height: int, width: int, boundary: str = "periodic", seed: Optional[int] = None):

        self.height = height
        self.width = width
        self.boundary = boundary
        self.rng = np.random.default_rng(seed)


        self.grid = np.zeros((4, height, width), dtype=np.uint8)
        

        self.walls = np.zeros((height, width), dtype=bool)
        if self.boundary == "reflective":
            self.walls[0, :] = True
            self.walls[-1, :] = True
            self.walls[:, 0] = True
            self.walls[:, -1] = True

    def reset(self) -> None:
        self.grid.fill(0)

    def add_obstacle(self, y_slice: slice, x_slice: slice) -> None:
        self.walls[y_slice, x_slice] = True
        self.grid[:, y_slice, x_slice] = 0

    def init_random(self, density: float) -> None:
        raw = self.rng.random((4, self.height, self.width)) < density
        self.grid = raw.astype(np.uint8)
        self.grid[:, self.walls] = 0

    def init_disk(self, center_y: int, center_x: int, radius: int, density: float = 0.5) -> None:
        self.reset()
        y, x = np.ogrid[:self.height, :self.width]
        mask = ((y - center_y) ** 2 + (x - center_x) ** 2) <= radius ** 2
        for i in range(4):
            particle_mask = (self.rng.random((self.height, self.width)) < density) & mask
            self.grid[i, particle_mask] = 1
        self.grid[:, self.walls] = 0

    def collide(self) -> None:
        n0 = self.grid[0].astype(np.int8)
        n1 = self.grid[1].astype(np.int8)
        n2 = self.grid[2].astype(np.int8)
        n3 = self.grid[3].astype(np.int8)

        head_on_02 = n0 * n2 * (1 - n1) * (1 - n3)
        head_on_13 = n1 * n3 * (1 - n0) * (1 - n2)
        delta = head_on_02 - head_on_13

        self.grid[0] = (n0 - delta).astype(np.uint8)
        self.grid[2] = (n2 - delta).astype(np.uint8)
        self.grid[1] = (n1 + delta).astype(np.uint8)
        self.grid[3] = (n3 + delta).astype(np.uint8)

        if np.any(self.walls):
            self.grid[:, self.walls] = 0

    def stream(self) -> None:

        new_grid = np.zeros_like(self.grid)

        # v0: вправо (сдвиг по столбцам +1)
        new_grid[0] = np.roll(self.grid[0], shift=1, axis=1)
        # v1: вверх (сдвиг по строкам -1)
        new_grid[1] = np.roll(self.grid[1], shift=-1, axis=0)
        # v2: влево (сдвиг по столбцам -1)
        new_grid[2] = np.roll(self.grid[2], shift=-1, axis=1)
        # v3: вниз (сдвиг по строкам +1)
        new_grid[3] = np.roll(self.grid[3], shift=1, axis=0)


        if np.any(self.walls):
            hit_0 = new_grid[0] * self.walls
            hit_1 = new_grid[1] * self.walls
            hit_2 = new_grid[2] * self.walls
            hit_3 = new_grid[3] * self.walls

            new_grid[0][self.walls] = 0
            new_grid[1][self.walls] = 0
            new_grid[2][self.walls] = 0
            new_grid[3][self.walls] = 0


            new_grid[2] += np.roll(hit_0, shift=-1, axis=1)
            new_grid[3] += np.roll(hit_1, shift=1, axis=0)
            new_grid[0] += np.roll(hit_2, shift=1, axis=1)
            new_grid[1] += np.roll(hit_3, shift=-1, axis=0)

        self.grid = new_grid

    def step(self) -> None:
        self.collide()
        self.stream()

    def reverse_velocities(self) -> None:
        self.grid[[0, 2]] = self.grid[[2, 0]]
        self.grid[[1, 3]] = self.grid[[3, 1]]

    @property
    def density(self) -> np.ndarray:
        return np.sum(self.grid, axis=0)

    @property
    def total_mass(self) -> int:
        return int(np.sum(self.grid))

    @property
    def total_momentum(self) -> Tuple[float, float]:
        jx = int(np.sum(self.grid[0]) - np.sum(self.grid[2]))
        # В декартовых координатах y направлен вверх (v1=+1, v3=-1)
        jy = int(np.sum(self.grid[1]) - np.sum(self.grid[3]))
        return float(jx), float(jy)