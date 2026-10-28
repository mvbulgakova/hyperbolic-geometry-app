# Гиперболическая сфера

Интерактивная визуализация гиперболической сферы в модели Бельтрами–Клейна. Сфера задаётся центром (три ползунка, от −0.6 до 0.6 по каждой оси) и евклидовым радиусом (от 0.05 до 0.8). Оси координат скрываются кнопкой.

## Запуск

```
pip install -r requirements.txt
python hyperbolic_sphere_app.py
```

Приложение открывается на http://127.0.0.1:8050.

На сервере оно запускается через gunicorn, как в `Procfile`:

```
gunicorn hyperbolic_sphere_app:server --bind 0.0.0.0:8000
```

## Python 3.14

Проверено в Termux: на Python 3.14 с Dash 3.1.1 запуск через `python hyperbolic_sphere_app.py` падает с ошибкой `pkgutil.find_loader` (режим `debug=True`). Запускать так:

```
python -c "from hyperbolic_sphere_app import app; app.run(debug=False)"
```

## Termux

Свежий Dash 4.x в Termux не ставится: ему нужен `pydantic-core`, который собирается из исходников на Rust. Рабочий набор:

```
pkg install python-numpy
pip install dash==3.1.1 plotly==6.2.0 Flask==3.1.1
```

## Другие версии

Версия с тремя вкладками (сфера, орисфера, эквидистанта) сохранена в ветке `merged-3-tabs`. Орисфера и эквидистанта вынесены в отдельные репозитории: `hyperbolic-geometry2-app` и `hyperbolic-geometry3-app`.

## Курсовая работа

Текст курсовой (LaTeX), рисунки и собранный черновик лежат в папке [`kursovaya/`](kursovaya/).
