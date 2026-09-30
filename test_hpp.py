"""
test_hpp.py
Автоматическое тестирование модели HPP с помощью pytest.
"""

import pytest
import numpy as np
from model import HPPModel


def test_collision_rules_head_on_02():
    """Лобовое столкновение по горизонтали (0 и 2) должно рассеяться в (1 и 3)."""
    model = HPPModel(height=3, width=3)
    model.grid[0, 1, 1] = 1
    model.grid[2, 1, 1] = 1
    model.collide()
    assert model.grid[0, 1, 1] == 0
    assert model.grid[2, 1, 1] == 0
    assert model.grid[1, 1, 1] == 1
    assert model.grid[3, 1, 1] == 1


def test_collision_rules_head_on_13():
    """Лобовое столкновение по вертикали (1 и 3) должно рассеяться в (0 и 2)."""
    model = HPPModel(height=3, width=3)
    model.grid[1, 1, 1] = 1
    model.grid[3, 1, 1] = 1
    model.collide()
    assert model.grid[1, 1, 1] == 0
    assert model.grid[3, 1, 1] == 0
    assert model.grid[0, 1, 1] == 1
    assert model.grid[2, 1, 1] == 1


def test_collision_perpendicular_no_interaction():
    """Перпендикулярные частицы (0 и 1) пролетают без взаимодействия."""
    model = HPPModel(height=3, width=3)
    model.grid[0, 1, 1] = 1
    model.grid[1, 1, 1] = 1
    model.collide()
    assert model.grid[0, 1, 1] == 1
    assert model.grid[1, 1, 1] == 1
    assert model.grid[2, 1, 1] == 0
    assert model.grid[3, 1, 1] == 0


def test_collision_three_particles_no_interaction():
    """Три частицы в узле не сталкиваются (согласно правилам HPP)."""
    model = HPPModel(height=3, width=3)
    model.grid[0, 1, 1] = 1
    model.grid[1, 1, 1] = 1
    model.grid[2, 1, 1] = 1
    model.collide()
    assert model.grid[0, 1, 1] == 1
    assert model.grid[1, 1, 1] == 1
    assert model.grid[2, 1, 1] == 1
    assert model.grid[3, 1, 1] == 0


def test_collision_four_particles_no_interaction():
    """Четыре частицы в узле пролетают без изменения."""
    model = HPPModel(height=3, width=3)
    model.grid[:, 1, 1] = 1
    model.collide()
    assert np.all(model.grid[:, 1, 1] == 1)


def test_mass_conservation_periodic():
    """Проверка строгого сохранения массы при периодических границах."""
    model = HPPModel(height=20, width=20, boundary="periodic", seed=42)
    model.init_random(density=0.3)
    initial_mass = model.total_mass

    for _ in range(50):
        model.step()
        assert model.total_mass == initial_mass


def test_momentum_conservation_periodic():
    """Проверка строгого сохранения импульса при периодических границах."""
    model = HPPModel(height=25, width=25, boundary="periodic", seed=10)
    model.init_random(density=0.4)
    init_jx, init_jy = model.total_momentum

    for _ in range(50):
        model.step()
        curr_jx, curr_jy = model.total_momentum
        assert curr_jx == init_jx
        assert curr_jy == init_jy


def test_time_reversibility():
    """Проверка строгой T-обратимости эволюции автомата."""
    model = HPPModel(height=15, width=15, boundary="periodic", seed=77)
    model.init_random(density=0.3)
    initial_state = model.grid.copy()

    steps = 25
    for _ in range(steps):
        model.step()

    model.reverse_velocities()

    for _ in range(steps):
        model.stream()
        model.collide()

    model.reverse_velocities()
    assert np.array_equal(model.grid, initial_state)