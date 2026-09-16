"""Reproducción didáctica del caso single-bus de Dias Garcia et al.

Objetivo
--------
Comparar el enfoque abierto LS-Ex con el enfoque cerrado Opt-Opt en un sistema
con 1 barra, 1 carga y 4 generadores.

Referencia principal
--------------------
J. Dias Garcia, A. Street, T. Homem-de-Mello, F. D. Muñoz,
"Application-Driven Learning: A Closed-Loop Prediction and Optimization Approach
Applied to Dynamic Reserves and Demand Forecasting", Operations Research 73(1).

Este script reproduce la ESTRUCTURA publicada del experimento, no intenta ser una
réplica bit-a-bit del código suplementario original.

Dependencias: numpy, scipy, pandas, matplotlib.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize


# -----------------------------------------------------------------------------
# 1. Sistema eléctrico del experimento ilustrativo
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SingleBusSystem:
    """Parámetros del sistema single-bus descrito en el paper."""

    capacity: np.ndarray
    gen_cost: np.ndarray
    reserve_fraction: float = 0.30
    reserve_cost_fraction: float = 0.30
    load_shed_multiplier: float = 8.0
    spill_multiplier: float = 3.0

    @classmethod
    def from_paper(cls) -> "SingleBusSystem":
        return cls(
            capacity=np.array([5.0, 5.0, 2.5, 2.5], dtype=float),
            gen_cost=np.array([1.0, 2.0, 4.0, 8.0], dtype=float),
        )

    @property
    def n_gen(self) -> int:
        return len(self.capacity)

    @property
    def reserve_cap(self) -> np.ndarray:
        return self.reserve_fraction * self.capacity

    @property
    def reserve_cost(self) -> np.ndarray:
        # El paper fija el costo de asignación de reserva en 30% del costo nominal.
        return self.reserve_cost_fraction * self.gen_cost

    @property
    def lambda_ls(self) -> float:
        # Déficit / load shedding = 8 x costo del generador más caro.
        return self.load_shed_multiplier * float(np.max(self.gen_cost))

    @property
    def lambda_sp(self) -> float:
        # Spillage / generación excedente = 3 x costo del generador más caro.
        return self.spill_multiplier * float(np.max(self.gen_cost))

    @property
    def total_reserve_cap(self) -> float:
        return float(np.sum(self.reserve_cap))


@dataclass
class Plan:
    generation: np.ndarray
    reserve_up: np.ndarray
    reserve_down: np.ndarray
    planned_load_shed: float
    planned_spill: float
    planning_objective: float


class Planner:
    """LP ex-ante del paper, especializado a una sola barra.

    Variables x = [g, r_up, r_down, delta_LS, delta_SP].

    min c'g + p_up'r_up + p_down'r_down
        + lambda_LS*delta_LS + lambda_SP*delta_SP

    s.a.
        sum(g) + delta_LS - delta_SP = D_hat
        sum(r_up)   = R_up_hat
        sum(r_down) = R_down_hat
        g + r_up <= K
        g - r_down >= 0
        r_up <= rbar_up
        r_down <= rbar_down
        variables >= 0

    Para una barra desaparecen las restricciones de flujo de red.
    """

    def __init__(self, system: SingleBusSystem):
        self.sys = system
        n = system.n_gen
        self.n_var = 3 * n + 2
        self.idx_ls = 3 * n
        self.idx_sp = 3 * n + 1

        self.c_obj = np.concatenate(
            [
                system.gen_cost,
                system.reserve_cost,
                system.reserve_cost,
                [system.lambda_ls, system.lambda_sp],
            ]
        )

        # Igualdades: balance de potencia y requisitos de reservas.
        self.A_eq = np.zeros((3, self.n_var))
        self.A_eq[0, :n] = 1.0
        self.A_eq[0, self.idx_ls] = 1.0
        self.A_eq[0, self.idx_sp] = -1.0
        self.A_eq[1, n : 2 * n] = 1.0
        self.A_eq[2, 2 * n : 3 * n] = 1.0

        # Desigualdades A_ub x <= b_ub.
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for i in range(n):
            # g_i + r_up_i <= K_i
            row = np.zeros(self.n_var)
            row[i] = 1.0
            row[n + i] = 1.0
            rows.append(row)
            rhs.append(system.capacity[i])

            # -g_i + r_down_i <= 0  <=> g_i - r_down_i >= 0
            row = np.zeros(self.n_var)
            row[i] = -1.0
            row[2 * n + i] = 1.0
            rows.append(row)
            rhs.append(0.0)

            # r_up_i <= rbar_i
            row = np.zeros(self.n_var)
            row[n + i] = 1.0
            rows.append(row)
            rhs.append(system.reserve_cap[i])

            # r_down_i <= rbar_i
            row = np.zeros(self.n_var)
            row[2 * n + i] = 1.0
            rows.append(row)
            rhs.append(system.reserve_cap[i])

        self.A_ub = np.vstack(rows)
        self.b_ub = np.asarray(rhs)

    def solve(self, demand_forecast: float, reserve_up: float, reserve_down: float) -> Plan:
        b_eq = np.array([demand_forecast, reserve_up, reserve_down], dtype=float)

        result = linprog(
            c=self.c_obj,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )

        if not result.success:
            raise RuntimeError(f"LP de planificación infactible: {result.message}")

        n = self.sys.n_gen
        x = result.x
        return Plan(
            generation=x[:n].copy(),
            reserve_up=x[n : 2 * n].copy(),
            reserve_down=x[2 * n : 3 * n].copy(),
            planned_load_shed=float(x[self.idx_ls]),
            planned_spill=float(x[self.idx_sp]),
            planning_objective=float(result.fun),
        )


# -----------------------------------------------------------------------------
# 2. Etapa ex-post / real-time
# -----------------------------------------------------------------------------


def actual_operation_cost(system: SingleBusSystem, plan: Plan, actual_demand: float) -> dict:
    """Evalúa el costo real Ga para una barra.

    En la formulación publicada, el costo de generación programada y de reserva
    asignada queda fijado por el plan. En tiempo real la generación puede moverse
    dentro de [g*-r_down, g*+r_up]. En una barra, si la demanda cae dentro de la
    suma de esos intervalos, existe un redespacho sin déficit ni spillage.

    Esto nos permite evaluar la segunda etapa analíticamente sin resolver otro LP.
    """

    lower = float(np.sum(plan.generation - plan.reserve_down))
    upper = float(np.sum(plan.generation + plan.reserve_up))

    load_shed = max(float(actual_demand) - upper, 0.0)
    spill = max(lower - float(actual_demand), 0.0)

    fixed_cost = float(
        system.gen_cost @ plan.generation
        + system.reserve_cost @ plan.reserve_up
        + system.reserve_cost @ plan.reserve_down
    )
    imbalance_cost = system.lambda_ls * load_shed + system.lambda_sp * spill

    return {
        "cost": fixed_cost + imbalance_cost,
        "fixed_cost": fixed_cost,
        "imbalance_cost": imbalance_cost,
        "load_shed": load_shed,
        "spill": spill,
        "lower_dispatch": lower,
        "upper_dispatch": upper,
    }


# -----------------------------------------------------------------------------
# 3. Datos sintéticos AR(1)
# -----------------------------------------------------------------------------


def generate_ar1_pairs(
    n: int,
    *,
    seed: int,
    theta0: float = 0.6,
    theta1: float = 0.9,
    long_run_mean: float = 6.0,
    coefficient_of_variation: float = 0.4,
    burn_in: int = 1000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Genera (D_{t-1}, D_t) siguiendo el experimento publicado.

    El paper especifica theta0=0.6, theta1=0.9, media de largo plazo 6,
    coeficiente de variación del proceso 0.4 y truncamiento de demandas negativas.
    No explicita en el texto la desviación estándar de la innovación epsilon.

    Aquí la inferimos imponiendo, ANTES DEL TRUNCAMIENTO, que el AR(1) gaussiano
    estacionario tenga std = CV*media. Para AR(1):
        sigma_D = sigma_eps / sqrt(1-theta1^2).
    Por tanto:
        sigma_eps = CV*media*sqrt(1-theta1^2).

    El truncamiento en cero modifica levemente los momentos efectivos del proceso.
    """

    rng = np.random.default_rng(seed)
    stationary_std = coefficient_of_variation * long_run_mean
    innovation_std = stationary_std * np.sqrt(1.0 - theta1**2)

    total = n + burn_in + 1
    d = np.empty(total, dtype=float)
    d[0] = long_run_mean

    for t in range(1, total):
        eps = rng.normal(0.0, innovation_std)
        d[t] = max(theta0 + theta1 * d[t - 1] + eps, 0.0)

    d = d[burn_in:]
    return d[:-1], d[1:], float(innovation_std)


def fit_least_squares(d_prev: np.ndarray, d_now: np.ndarray) -> tuple[np.ndarray, float]:
    X = np.column_stack([np.ones_like(d_prev), d_prev])
    theta, *_ = np.linalg.lstsq(X, d_now, rcond=None)
    residuals = d_now - X @ theta
    # Dos parámetros estimados: intercepto y coeficiente AR(1).
    dof = max(len(residuals) - 2, 1)
    sigma = float(np.sqrt(np.sum(residuals**2) / dof))
    return theta, sigma


# -----------------------------------------------------------------------------
# 4. Evaluación de una política forecast + reservas
# -----------------------------------------------------------------------------


def evaluate_policy(
    planner: Planner,
    d_prev: np.ndarray,
    d_now: np.ndarray,
    theta_d: np.ndarray,
    reserve_up: float,
    reserve_down: float,
    *,
    return_rows: bool = False,
) -> tuple[float, pd.DataFrame | None]:
    system = planner.sys
    rows = [] if return_rows else None
    total_cost = 0.0

    for prev, actual in zip(d_prev, d_now, strict=True):
        forecast = float(theta_d[0] + theta_d[1] * prev)
        plan = planner.solve(forecast, reserve_up, reserve_down)
        op = actual_operation_cost(system, plan, actual)
        total_cost += op["cost"]

        if return_rows:
            rows.append(
                {
                    "D_prev": prev,
                    "D_actual": actual,
                    "D_forecast": forecast,
                    "forecast_error_actual_minus_forecast": actual - forecast,
                    "cost": op["cost"],
                    "fixed_cost": op["fixed_cost"],
                    "imbalance_cost": op["imbalance_cost"],
                    "load_shed": op["load_shed"],
                    "spill": op["spill"],
                    "dispatch_lower": op["lower_dispatch"],
                    "dispatch_upper": op["upper_dispatch"],
                }
            )

    mean_cost = total_cost / len(d_now)
    return mean_cost, pd.DataFrame(rows) if return_rows else None


# -----------------------------------------------------------------------------
# 5. Entrenamiento application-driven con Nelder-Mead
# -----------------------------------------------------------------------------


def _reserve_penalty(system: SingleBusSystem, rup: float, rdn: float) -> float:
    """Penaliza requisitos de reserva fuera del conjunto obviamente factible."""
    cap = system.total_reserve_cap
    violation = max(-rup, 0.0) + max(-rdn, 0.0) + max(rup - cap, 0.0) + max(rdn - cap, 0.0)
    return 1e6 * violation if violation > 0.0 else 0.0


def optimize_application_driven(
    planner: Planner,
    d_prev: np.ndarray,
    d_now: np.ndarray,
    theta_ls: np.ndarray,
    reserve_ex: float,
    *,
    mode: str,
    maxiter: int,
    fatol: float = 1e-7,
) -> dict:
    """Entrena LS-Opt, Opt-Ex u Opt-Opt usando búsqueda derivative-free.

    Esto sigue la idea de la heurística del paper: para cada candidato theta,
    resolver el problema de planificación de cada observación y evaluar el costo
    ex-post en la realización observada.
    """

    system = planner.sys
    mode = mode.lower()
    if mode not in {"ls-opt", "opt-ex", "opt-opt"}:
        raise ValueError(f"Modo desconocido: {mode}")

    if mode == "ls-opt":
        x0 = np.array([reserve_ex, reserve_ex], dtype=float)
    elif mode == "opt-ex":
        x0 = theta_ls.astype(float).copy()
    else:
        x0 = np.array([theta_ls[0], theta_ls[1], reserve_ex, reserve_ex], dtype=float)

    cache: dict[tuple[float, ...], float] = {}
    n_eval = 0

    def decode(x: np.ndarray) -> tuple[np.ndarray, float, float]:
        if mode == "ls-opt":
            return theta_ls, float(x[0]), float(x[1])
        if mode == "opt-ex":
            return np.asarray(x[:2], dtype=float), reserve_ex, reserve_ex
        return np.asarray(x[:2], dtype=float), float(x[2]), float(x[3])

    def objective(x: np.ndarray) -> float:
        nonlocal n_eval
        # Redondeo solo para cachear evaluaciones numéricamente repetidas.
        key = tuple(np.round(np.asarray(x, dtype=float), 12))
        if key in cache:
            return cache[key]

        theta_d, rup, rdn = decode(np.asarray(x, dtype=float))
        penalty = _reserve_penalty(system, rup, rdn)
        if penalty > 0:
            val = 1e5 + penalty
        else:
            try:
                val, _ = evaluate_policy(
                    planner, d_prev, d_now, theta_d, rup, rdn, return_rows=False
                )
            except RuntimeError:
                val = 1e8

        cache[key] = float(val)
        n_eval += 1
        return float(val)

    t0 = time.perf_counter()
    res = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        options={
            "maxiter": maxiter,
            "fatol": fatol,
            "xatol": 1e-7,
            "adaptive": False,
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - t0

    theta_d, rup, rdn = decode(res.x)
    return {
        "mode": mode,
        "theta_d": np.asarray(theta_d, dtype=float),
        "reserve_up": float(rup),
        "reserve_down": float(rdn),
        "train_cost": float(res.fun),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "nfev_scipy": int(res.nfev),
        "nfev_unique": int(n_eval),
        "elapsed_s": float(elapsed),
    }


# -----------------------------------------------------------------------------
# 6. Experimento completo
# -----------------------------------------------------------------------------


def save_plots(results_dir: Path, test_tables: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Costo out-of-sample.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(summary["model"], summary["test_cost"])
    ax.set_ylabel("Costo operacional promedio out-of-sample")
    ax.set_title("Single-bus: comparación de políticas")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(results_dir / "01_out_of_sample_cost.png", dpi=160)
    plt.close(fig)

    # Histograma de error: LS-Ex vs Opt-Opt, si están disponibles.
    if "LS-Ex" in test_tables and "Opt-Opt" in test_tables:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name in ["LS-Ex", "Opt-Opt"]:
            err = test_tables[name]["forecast_error_actual_minus_forecast"]
            ax.hist(err, bins=40, alpha=0.55, density=True, label=name)
        ax.axvline(0.0, linewidth=1.0)
        ax.set_xlabel("Error = realización - forecast")
        ax.set_ylabel("Densidad")
        ax.set_title("Sesgo del forecast")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "02_forecast_error_histogram.png", dpi=160)
        plt.close(fig)

    # Demandas reales y forecast para una ventana corta.
    if "LS-Ex" in test_tables and "Opt-Opt" in test_tables:
        n_show = min(120, len(test_tables["LS-Ex"]))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(test_tables["LS-Ex"]["D_actual"].iloc[:n_show].to_numpy(), label="Real")
        ax.plot(test_tables["LS-Ex"]["D_forecast"].iloc[:n_show].to_numpy(), label="LS-Ex")
        ax.plot(test_tables["Opt-Opt"]["D_forecast"].iloc[:n_show].to_numpy(), label="Opt-Opt")
        ax.set_xlabel("Observación")
        ax.set_ylabel("Demanda")
        ax.set_title("Forecast estadístico vs application-driven")
        ax.legend()
        fig.tight_layout()
        fig.savefig(results_dir / "03_forecasts_window.png", dpi=160)
        plt.close(fig)


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    system = SingleBusSystem.from_paper()
    planner = Planner(system)

    train_prev, train_now, innovation_std = generate_ar1_pairs(
        args.train_size, seed=args.seed
    )
    test_prev, test_now, _ = generate_ar1_pairs(
        args.test_size, seed=args.seed + 100_000
    )

    theta_ls, residual_std = fit_least_squares(train_prev, train_now)
    reserve_ex = 1.96 * residual_std

    if reserve_ex > system.total_reserve_cap:
        warnings.warn(
            "La regla exógena 1.96*sigma supera la capacidad total de reserva. "
            "Se recorta para mantener factibilidad del LP."
        )
        reserve_ex = system.total_reserve_cap

    print("\n=== Datos y sistema ===")
    print(f"Train size              : {args.train_size}")
    print(f"Test size               : {args.test_size}")
    print(f"Innovación sigma usada  : {innovation_std:.6f}")
    print(f"Media train observada   : {train_now.mean():.4f}")
    print(f"Std train observada     : {train_now.std(ddof=1):.4f}")
    print(f"LS theta0, theta1       : {theta_ls[0]:.6f}, {theta_ls[1]:.6f}")
    print(f"LS sigma residual       : {residual_std:.6f}")
    print(f"Reserva exógena ±1.96σ  : {reserve_ex:.6f}")
    print(f"Cap. reserva total      : {system.total_reserve_cap:.4f}\n")

    policies: dict[str, dict] = {
        "LS-Ex": {
            "theta_d": theta_ls,
            "reserve_up": reserve_ex,
            "reserve_down": reserve_ex,
            "train_cost": np.nan,
            "elapsed_s": 0.0,
            "success": True,
        }
    }

    modes = ["opt-opt"] if not args.all_models else ["ls-opt", "opt-ex", "opt-opt"]
    display_name = {"ls-opt": "LS-Opt", "opt-ex": "Opt-Ex", "opt-opt": "Opt-Opt"}

    for mode in modes:
        print(f"Entrenando {display_name[mode]} ...")
        trained = optimize_application_driven(
            planner,
            train_prev,
            train_now,
            theta_ls,
            reserve_ex,
            mode=mode,
            maxiter=args.maxiter,
        )
        policies[display_name[mode]] = trained
        print(
            f"  costo train={trained['train_cost']:.6f}, "
            f"theta={trained['theta_d']}, "
            f"Rup={trained['reserve_up']:.4f}, Rdn={trained['reserve_down']:.4f}, "
            f"time={trained['elapsed_s']:.1f}s"
        )
        if not trained["success"]:
            print(f"  aviso Nelder-Mead: {trained['message']}")

    # Evaluación train/test para todos los métodos.
    rows = []
    test_tables: dict[str, pd.DataFrame] = {}
    for name, pol in policies.items():
        theta = np.asarray(pol["theta_d"], dtype=float)
        rup = float(pol["reserve_up"])
        rdn = float(pol["reserve_down"])
        train_cost, _ = evaluate_policy(
            planner, train_prev, train_now, theta, rup, rdn, return_rows=False
        )
        test_cost, test_df = evaluate_policy(
            planner, test_prev, test_now, theta, rup, rdn, return_rows=True
        )
        test_tables[name] = test_df
        rows.append(
            {
                "model": name,
                "theta0": theta[0],
                "theta1": theta[1],
                "reserve_up": rup,
                "reserve_down": rdn,
                "train_cost": train_cost,
                "test_cost": test_cost,
                "test_rmse": float(
                    np.sqrt(np.mean((test_df["D_actual"] - test_df["D_forecast"]) ** 2))
                ),
                "test_mean_error_actual_minus_forecast": float(
                    test_df["forecast_error_actual_minus_forecast"].mean()
                ),
                "test_mean_load_shed": float(test_df["load_shed"].mean()),
                "test_mean_spill": float(test_df["spill"].mean()),
                "optimization_time_s": float(pol.get("elapsed_s", 0.0)),
            }
        )

    summary = pd.DataFrame(rows).sort_values("test_cost").reset_index(drop=True)
    base = float(summary.loc[summary["model"] == "LS-Ex", "test_cost"].iloc[0])
    summary["improvement_vs_LS_Ex_pct"] = 100.0 * (base - summary["test_cost"]) / base

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "summary.csv", index=False)
    for name, table in test_tables.items():
        safe = name.lower().replace("-", "_")
        table.to_csv(results_dir / f"test_{safe}.csv", index=False)
    save_plots(results_dir, test_tables, summary)

    print("\n=== Resultados out-of-sample ===")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    print(f"\nResultados guardados en: {results_dir.resolve()}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Además de LS-Ex y Opt-Opt, entrena LS-Opt y Opt-Ex.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/single_bus",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
