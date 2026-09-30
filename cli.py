import click
import numpy as np
import matplotlib.pyplot as plt
from model import HPPModel


@click.group()
def cli():
    """Симулятор решёточного газа HPP D2Q4."""
    pass


@cli.command()
@click.option("--size", default=101, help="Размер квадратной решётки (NxN).")
@click.option("--radius", default=15, help="Радиус начального круглого облака газа.")
@click.option("--steps", default=120, help="Количество тактов симуляции.")
@click.option("--output", default="anisotropy_experiment.png", help="Путь для сохранения графиков.")
def run_cloud(size: int, radius: int, steps: int, output: str):
    """
    Эксперимент 1: Эволюция круглого облака газа.
    Демонстрация ромбовидной анизотропии сетки D2Q4.
    """
    click.echo(f"[*] Запуск эволюции облака: решётка {size}x{size}, радиус={radius}, шагов={steps}")
    model = HPPModel(height=size, width=size, boundary="periodic", seed=42)
    model.init_disk(center_y=size // 2, center_x=size // 2, radius=radius, density=0.7)

    snapshots = [0, steps // 4, steps // 2, steps]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    current_step = 0
    snap_idx = 0

    for current_step in range(steps + 1):
        if current_step in snapshots:
            ax = axes[snap_idx]
            im = ax.imshow(model.density, cmap="inferno", vmin=0, vmax=4)
            ax.set_title(f"t = {current_step}")
            ax.set_xticks([])
            ax.set_yticks([])
            snap_idx += 1
        model.step()

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="Плотность (число частиц)")
    plt.suptitle("Эволюция плотности в HPP D2Q4: проявление ромбовидной анизотропии", fontsize=14)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    click.echo(f"[+] График сохранён в {output}")


@cli.command()
@click.option("--size", default=50, help="Размер квадратной решётки.")
@click.option("--steps", default=200, help="Число шагов.")
@click.option("--output", default="conservation_test.png", help="Путь для графика сохранения.")
def test_conservation(size: int, steps: int, output: str):
    """
    Эксперимент 2: Численная проверка сохранения массы и импульса.
    """
    click.echo(f"[*] Проверка законов сохранения: {steps} шагов...")
    model = HPPModel(height=size, width=size, boundary="periodic", seed=123)
    model.init_random(density=0.25)

    masses = []
    mom_x = []
    mom_y = []

    for t in range(steps):
        masses.append(model.total_mass)
        jx, jy = model.total_momentum
        mom_x.append(jx)
        mom_y.append(jy)
        model.step()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(masses, color="blue", lw=2, label="Полная масса (N)")
    ax1.set_ylabel("Масса (число частиц)")
    ax1.set_title("Проверка закона сохранения массы")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.plot(mom_x, color="red", label="Импульс Jx")
    ax2.plot(mom_y, color="green", label="Импульс Jy")
    ax2.set_xlabel("Такт времени t")
    ax2.set_ylabel("Импульс")
    ax2.set_title("Проверка закона сохранения импульса")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    click.echo(f"[+] График законов сохранения сохранён в {output}")
    click.echo(f"    Дельта массы: max - min = {max(masses) - min(masses)}")
    click.echo(f"    Дельта Jx: max - min = {max(mom_x) - min(mom_x)}")


@cli.command()
@click.option("--size", default=50, help="Размер решётки.")
@click.option("--forward-steps", default=40, help="Шагов вперёд до обращения времени.")
def test_reversibility(size: int, forward_steps: int):
    """
    Эксперимент 3: Проверка T-обратимости модели HPP.
    """
    click.echo(f"[*] Моделирование {forward_steps} шагов вперёд, затем разворот скоростей...")
    model = HPPModel(height=size, width=size, boundary="periodic", seed=999)
    model.init_disk(center_y=size // 2, center_x=size // 2, radius=8, density=0.6)
    
    initial_state = model.grid.copy()

    # Шаги вперёд
    for _ in range(forward_steps):
        model.step()

    # Обращение скоростей
    model.reverse_velocities()

    # Шаги назад (эволюция с развёрнутыми скоростями)
    for _ in range(forward_steps):
        # При обратном ходе: сначала перенос, потом столкновение (сопряжённый оператор)
        model.stream()
        model.collide()

    # Разворачиваем снова для прямого сравнения исходных направлений
    model.reverse_velocities()

    diff = np.sum(np.abs(model.grid.astype(int) - initial_state.astype(int)))
    click.echo(f"[+] Расхождение с начальным состоянием: {diff} ячеек")
    if diff == 0:
        click.echo("    УСПЕХ: Модель HPP строго обратима во времени (T-инвариантна)!")
    else:
        click.echo("    ОШИБКА: Обратимость нарушена.")


if __name__ == "__main__":
    cli()