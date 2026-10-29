"""Практичне заняття 4. Прогнозування часового ряду.

Варіант 2: прогнозування з урахуванням сезонності.
Модель SARIMA (з сезонними параметрами P, D, Q) порівнюється з моделлю ARIMA без сезонності
на штучному ряді місячних продажів (тренд + річна сезонність + випадкові коливання).

Запуск:
    python forecasting.py                      діалог вибору моделі й горизонту
    python forecasting.py --model both --horizon 12 [--plot file.png]
"""

import argparse
import sys
import warnings

import matplotlib

matplotlib.use("Agg")  # графіки зберігаються у файл, вікно не потрібне

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

SEASON = 12                      # довжина сезону: 12 місяців
ARIMA_ORDER = (2, 1, 2)          # (p, d, q)
SEASONAL_ORDER = (1, 1, 1, SEASON)  # (P, D, Q, s)
HORIZONS = (6, 12, 24)           # горизонти прогнозу, з яких можна обирати
MODELS = {"1": "arima", "2": "sarima", "3": "both"}


# 1. Завантаження й підготовка даних -------------------------------------------------------------

def make_sales(months=96, seed=42):
    """Штучні місячні продажі: лінійний тренд + річна сезонність + шум."""
    rng = np.random.default_rng(seed)
    t = np.arange(months)
    trend = 200 + 2.5 * t
    # пік продажів у грудні, спад улітку
    seasonal = 40 * np.sin(2 * np.pi * (t - 9) / SEASON) + 25 * (t % SEASON == 11)
    noise = rng.normal(0, 8, months)
    index = pd.date_range("2017-01-01", periods=months, freq="MS")
    return pd.Series(trend + seasonal + noise, index=index, name="sales")


# 2-3. Аналіз ряду й стаціонарність --------------------------------------------------------------

def adf_pvalue(series):
    """p-значення тесту Дікі-Фуллера: < 0.05 означає, що ряд стаціонарний."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # попередження про майбутню зміну типу результату statsmodels
        return adfuller(series.dropna())[1]


def analyse(series):
    """Перевіряє стаціонарність до й після диференціювання (звичайного і сезонного)."""
    return {
        "вихідний ряд": adf_pvalue(series),
        "після diff(1)": adf_pvalue(series.diff()),
        "після diff(1) і сезонного diff(12)": adf_pvalue(series.diff().diff(SEASON)),
    }


# 4-5. Побудова моделей і прогноз ----------------------------------------------------------------

def fit_forecast(train, horizon, seasonal):
    """Навчає ARIMA або SARIMA на train і повертає прогноз на horizon місяців."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # statsmodels попереджає про збіжність, на результат це не впливає
        if seasonal:
            model = SARIMAX(train, order=(1, 1, 1), seasonal_order=SEASONAL_ORDER)
        else:
            model = ARIMA(train, order=ARIMA_ORDER)
        fitted = model.fit(**({"disp": False} if seasonal else {}))
        return fitted.forecast(steps=horizon)


# 6. Оцінка точності -----------------------------------------------------------------------------

def evaluate(actual, forecast):
    """Середньоквадратична (MSE) і середня абсолютна (MAE) похибки прогнозу."""
    errors = np.asarray(actual) - np.asarray(forecast)
    return {"MSE": float(np.mean(errors ** 2)), "MAE": float(np.mean(np.abs(errors)))}


def run(model, horizon, series=None):
    """Повний цикл: дані -> навчання -> прогноз на останні horizon місяців -> метрики.

    Повертає словник {назва моделі: {"forecast": Series, "MSE": ..., "MAE": ...}} і дані.
    """
    if model not in MODELS.values():
        raise ValueError(f"Невідома модель: {model}")
    if horizon not in HORIZONS:
        raise ValueError(f"Горизонт має бути одним із {HORIZONS}")

    series = make_sales() if series is None else series
    train, test = series[:-horizon], series[-horizon:]

    kinds = {"arima": [False], "sarima": [True], "both": [False, True]}[model]
    results = {}
    for seasonal in kinds:
        name = f"SARIMA{(1, 1, 1)}x{SEASONAL_ORDER}" if seasonal else f"ARIMA{ARIMA_ORDER}"
        forecast = fit_forecast(train, horizon, seasonal)
        forecast.index = test.index
        results[name] = {"forecast": forecast, **evaluate(test, forecast)}
    return results, train, test


def plot(results, train, test, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(train.index, train, color="#555", label="навчальні дані")
    ax.plot(test.index, test, color="black", linewidth=2, label="фактичні значення")
    for (name, result), color in zip(results.items(), ("#d1495b", "#2e86ab")):
        ax.plot(result["forecast"].index, result["forecast"], "--", color=color, linewidth=2,
                label=f"{name}: MAE={result['MAE']:.1f}")
    ax.axvline(test.index[0], color="#999", linestyle=":")
    ax.set_title(f"Прогноз місячних продажів на {len(test)} міс.")
    ax.set_xlabel("місяць")
    ax.set_ylabel("продажі, од.")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def report(results, test):
    lines = []
    table = pd.DataFrame({"факт": test.round(1)})
    for name, result in results.items():
        table[name.split("(")[0]] = result["forecast"].round(1)
    table.index = table.index.strftime("%Y-%m")
    lines.append(table.to_string())
    lines.append("")
    for name, result in results.items():
        lines.append(f"{name:32} MSE = {result['MSE']:9.2f}   MAE = {result['MAE']:7.2f}")
    if len(results) == 2:
        (a_name, a), (s_name, s) = results.items()
        better = s_name if s["MAE"] < a["MAE"] else a_name
        lines.append(f"Точніша модель: {better} (MAE менша в {max(a['MAE'], s['MAE']) / min(a['MAE'], s['MAE']):.1f} раза)")
    return "\n".join(lines)


# Діалог з користувачем --------------------------------------------------------------------------

def ask(prompt, valid):
    """Питає, доки не буде введено одне з допустимих значень."""
    while True:
        try:
            answer = input(prompt).strip()
        except EOFError:
            print("\nВведення завершено.")
            sys.exit(0)
        if answer in valid:
            return answer
        print(f"Невірне значення. Допустимо: {', '.join(valid)}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARIMA vs SARIMA на сезонному часовому ряді")
    parser.add_argument("--model", choices=list(MODELS.values()))
    parser.add_argument("--horizon", type=int, choices=HORIZONS)
    parser.add_argument("--plot", default="forecast.png", help="файл для графіка")
    args = parser.parse_args(argv)

    series = make_sales()
    print("Дані: штучні місячні продажі, "
          f"{series.index[0]:%Y-%m} … {series.index[-1]:%Y-%m}, {len(series)} точок")
    print("Тест Дікі-Фуллера (p < 0.05 — ряд стаціонарний):")
    for step, p in analyse(series).items():
        print(f"  {step:38} p = {p:.4f}  {'стаціонарний' if p < 0.05 else 'нестаціонарний'}")

    model = args.model or MODELS[ask("\nМодель: 1 - ARIMA, 2 - SARIMA, 3 - порівняти обидві: ", list(MODELS))]
    horizon = args.horizon or int(ask(f"Горизонт прогнозу, місяців {HORIZONS}: ", [str(h) for h in HORIZONS]))

    results, train, test = run(model, horizon, series)
    print()
    print(report(results, test))
    plot(results, train, test, args.plot)
    print(f"\nГрафік збережено у {args.plot}")


if __name__ == "__main__":
    main()
