"""Тестування forecasting.py: усі варіанти вибору (модель x горизонт) і перевірка введення.

Результат: test_report.html з таблицями й графіками; PDF робиться з нього браузером Edge.
"""

import base64
import html
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import forecasting as f

HERE = Path(__file__).parent
PLOTS = HERE / "test_plots"
PLOTS.mkdir(exist_ok=True)
PY = sys.executable

sections = []
checks = []  # (назва перевірки, очікування, результат, пройдено?)


def check(name, expected, actual, passed):
    checks.append((name, expected, actual, passed))


# 1. Аналіз стаціонарності
series = f.make_sales()
adf = f.analyse(series)
check("Вихідний ряд нестаціонарний (тренд)", "p ≥ 0.05", f"p = {adf['вихідний ряд']:.4f}", adf["вихідний ряд"] >= 0.05)
check("Після диференціювання ряд стаціонарний", "p < 0.05", f"p = {adf['після diff(1)']:.4f}", adf["після diff(1)"] < 0.05)

# 2. Усі комбінації модель x горизонт
summary = []
for model in ("arima", "sarima", "both"):
    for horizon in f.HORIZONS:
        results, train, test = f.run(model, horizon, series)
        plot_path = PLOTS / f"{model}_{horizon}.png"
        f.plot(results, train, test, plot_path)
        expected_models = 2 if model == "both" else 1
        ok = (len(results) == expected_models
              and all(len(r["forecast"]) == horizon for r in results.values())
              and all(r["MSE"] >= r["MAE"] ** 2 - 1e-9 for r in results.values()))  # MSE ≥ MAE² завжди
        check(f"Модель {model}, горизонт {horizon}", f"{expected_models} прогноз(и) по {horizon} міс., коректні метрики",
              ", ".join(f"{n.split('(')[0]}: MAE {r['MAE']:.2f}" for n, r in results.items()), ok)
        for name, r in results.items():
            summary.append((model, horizon, name, r["MSE"], r["MAE"]))
        sections.append((f"Модель: {model}, горизонт: {horizon} міс.", f.report(results, test), plot_path))

# 3. SARIMA точніша за ARIMA на сезонному ряді
for horizon in f.HORIZONS:
    a = next(s for s in summary if s[0] == "both" and s[1] == horizon and s[2].startswith("ARIMA"))
    s = next(s for s in summary if s[0] == "both" and s[1] == horizon and s[2].startswith("SARIMA"))
    check(f"SARIMA точніша за ARIMA, {horizon} міс.", "MAE SARIMA < MAE ARIMA",
          f"{s[4]:.2f} < {a[4]:.2f}", s[4] < a[4])

# 4. Перевірка введення в діалозі
def dialog(inputs):
    out = io.StringIO()
    # після останньої відповіді введення "закінчується", як при Ctrl+Z
    with mock.patch("builtins.input", side_effect=list(inputs) + [EOFError()]), redirect_stdout(out):
        try:
            f.main(["--plot", str(PLOTS / "dialog.png")])
        except SystemExit:
            pass
    return out.getvalue()

text = dialog(["5", "abc", "", "3", "7", "0", "12"])
check("Невірна модель (5, abc, порожньо) -> повторний запит", "3 повідомлення про помилку, потім прийнято 3",
      f"{text.count('Невірне значення. Допустимо: 1, 2, 3')} повідомлення", text.count("Допустимо: 1, 2, 3") == 3)
check("Невірний горизонт (7, 0) -> повторний запит", "2 повідомлення, потім прийнято 12",
      f"{text.count('Допустимо: 6, 12, 24')} повідомлення", text.count("Допустимо: 6, 12, 24") == 2 and "Точніша модель" in text)
sections.append(("Діалог із невірним введенням: 5, abc, (порожньо), 3, 7, 0, 12", text, None))

text = dialog([])
check("Кінець введення (Ctrl+Z / EOF)", "коректне завершення без помилки", "«Введення завершено»", "Введення завершено" in text)

for value, name in (("1", "ARIMA"), ("2", "SARIMA")):
    text = dialog([value, "6"])
    check(f"Вибір у діалозі: {value} -> {name}", f"прогноз лише {name}", "одна модель у звіті",
          text.count("MSE =") == 1 and f"{name}(" in text)

# 5. Аргументи командного рядка
proc = subprocess.run([PY, str(HERE / "forecasting.py"), "--horizon", "7"], capture_output=True, text=True, encoding="utf-8")
check("Аргумент --horizon 7", "відмова argparse, код 2", f"код {proc.returncode}", proc.returncode == 2)
proc = subprocess.run([PY, str(HERE / "forecasting.py"), "--model", "lstm"], capture_output=True, text=True, encoding="utf-8")
check("Аргумент --model lstm", "відмова argparse, код 2", f"код {proc.returncode}", proc.returncode == 2)
for bad in (("svm", 6), ("both", 5)):
    try:
        f.run(*bad)
        ok = False
    except ValueError:
        ok = True
    check(f"run{bad}", "ValueError", "ValueError" if ok else "немає помилки", ok)

# Звіт ------------------------------------------------------------------------------------------
def img(path):
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}">'

passed = sum(c[3] for c in checks)
rows = "".join(
    f"<tr><td>{i}</td><td>{html.escape(n)}</td><td>{html.escape(e)}</td><td>{html.escape(a)}</td>"
    f"<td class={'ok' if p else 'bad'}>{'так' if p else 'НІ'}</td></tr>"
    for i, (n, e, a, p) in enumerate(checks, 1))
summary_rows = "".join(
    f"<tr><td>{m}</td><td>{h}</td><td>{html.escape(n)}</td><td>{mse:.2f}</td><td>{mae:.2f}</td></tr>"
    for m, h, n, mse, mae in summary if m == "both")
adf_rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td><td>{'так' if v < 0.05 else 'ні'}</td></tr>" for k, v in adf.items())
blocks = "".join(
    f"<div class=case><h3>{html.escape(t)}</h3><pre>{html.escape(r)}</pre>{img(p) if p else ''}</div>"
    for t, r, p in sections)

page = f"""<!doctype html><html lang=uk><meta charset=utf-8><title>Звіт тестування</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#222;font-size:13px}}
h1{{font-size:20px;margin-bottom:2px}} h2{{font-size:16px;margin-top:26px;border-bottom:1px solid #ccc}}
h3{{font-size:14px;margin:14px 0 6px}}
table{{border-collapse:collapse;margin:8px 0}} td,th{{border:1px solid #bbb;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#eee}} .ok{{color:#1a7f37;font-weight:600}} .bad{{color:#c00;font-weight:700}}
pre{{background:#f6f6f6;padding:8px;font-size:11.5px;white-space:pre-wrap}} img{{width:100%;max-width:820px}}
.case{{page-break-inside:avoid}}
</style>
<h1>Практичне заняття 4. Прогнозування часового ряду</h1>
<div>Варіант 2 — прогнозування з урахуванням сезонності (SARIMA проти ARIMA). Ращенко Тимофій, F511.</div>
<h2>Опис</h2>
<p>Дані: штучні місячні продажі за 2017–2024 роки (96 точок): лінійний тренд, річна сезонність із піком у грудні
та випадковий шум. Останні N місяців (N = 6, 12, 24) відкладаються як тестові, модель навчається на решті.
Моделі: ARIMA{f.ARIMA_ORDER} без сезонності та SARIMA(1, 1, 1)x{f.SEASONAL_ORDER} із сезонними параметрами
(P, D, Q) = (1, 1, 1), s = 12. Точність — MSE і MAE.</p>
<h2>Аналіз стаціонарності (тест Дікі-Фуллера)</h2>
<table><tr><th>Ряд</th><th>p-значення</th><th>Стаціонарний</th></tr>{adf_rows}</table>
<h2>Підсумок перевірок: {passed} з {len(checks)} пройдено</h2>
<table><tr><th>#</th><th>Перевірка</th><th>Очікування</th><th>Результат</th><th>OK</th></tr>{rows}</table>
<h2>Порівняння ARIMA і SARIMA</h2>
<table><tr><th>Режим</th><th>Горизонт</th><th>Модель</th><th>MSE</th><th>MAE</th></tr>{summary_rows}</table>
<p>Висновок: на ряді із сезонністю SARIMA дає меншу похибку на всіх горизонтах. Перевага зростає
з довжиною прогнозу, бо ARIMA без сезонних параметрів не може відтворити річний цикл на далеких кроках.</p>
<h2>Результати для всіх варіантів вибору</h2>
{blocks}
</html>"""

(HERE / "test_report.html").write_text(page, encoding="utf-8")
print(f"{passed}/{len(checks)} перевірок пройдено")
for n, e, a, p in checks:
    print(("OK  " if p else "FAIL"), n, "|", a)
