# ⚽ Football Signals Bot

Автономный Telegram-бот, который анализирует футбольные матчи с помощью ML
и публикует value-сигналы. Деплой на Railway в один клик.

## Что внутри

- **Данные**: [football-data.org](https://www.football-data.org/) (бесплатный API на топ-лиги)
- **Фичи**: Elo-рейтинг, форма за 10 матчей, голы дома/в гостях, очные встречи, отдых
- **Модель**: три калиброванных XGBoost-классификатора — 1X2, Тотал 2.5, Обе забьют
- **Сигналы**: value-bet логика (edge ≥ 5%) + размер ставки по 1/4 Kelly с капом 2 ед.
- **Бот**: aiogram 3 (асинхронный), команды `/signals`, `/today`, `/stats`, `/subscribe`
- **Трекинг**: каждая ставка settle-ится автоматически, ROI считается честно
- **Расписание**: APScheduler гоняет полный цикл раз в сутки, сигналы — каждые 3 часа

## ⚠️ Честно про прибыльность

Никто не гарантирует ROI на ставках. Букмекеры держат маржу 5–8%, и обыгрывать их стабильно — задача крайне сложная. Этот проект — **инструмент анализа**:

- Без фида котировок бот публикует только «MODEL»-сигналы (высокая уверенность модели) — это **не** value bets, а просто прогнозы.
- С котировками бот считает edge и публикует **VALUE**-сигналы. Реальная прибыльность зависит от качества линии, ликвидности и дисциплины.
- Команда `/stats` показывает фактический ROI **на основе твоих реальных сигналов**, без манипуляций.

Используй разумно, не ставь больше, чем готов потерять.

## Деплой на Railway

1. Создай бота у [@BotFather](https://t.me/BotFather), получи токен.
2. Получи бесплатный API-ключ на [football-data.org](https://www.football-data.org/client/register).
3. Форкни этот репо и подключи к Railway → New Project → Deploy from GitHub.
4. В Railway → Variables добавь:
   - `TELEGRAM_BOT_TOKEN`
   - `FOOTBALL_DATA_API_KEY`
   - `ADMIN_IDS` (твой Telegram user id)
   - (опционально) `COMPETITIONS=PL,PD,SA,BL1,FL1,CL`
5. Добавь Postgres плагин: Railway автоматически подкинет `DATABASE_URL`.
6. Жми Deploy. На первом запуске бот сам подтянет историю и обучит модель (~5–10 минут на free tier из-за лимита 10 req/min).

## Локальный запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # заполни TELEGRAM_BOT_TOKEN и FOOTBALL_DATA_API_KEY
python -m src.main
```

Тесты:

```bash
pip install pytest
pytest tests/
```

## Структура

```
src/
  bot/        # aiogram handlers, форматирование сообщений
  data/       # API клиент, БД-модели, ingest, feature engineering
  ml/         # обучение и инференс
  signals/    # генерация сигналов + ROI tracking
  pipeline.py # связка всего
  main.py     # точка входа, scheduler
```

## Конфигурация

Все настройки — через переменные окружения (см. `.env.example`):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MIN_EDGE` | 0.05 | Минимальный edge для value-сигнала |
| `MIN_CONFIDENCE` | 0.55 | Минимальная вероятность модели |
| `MIN_ODDS` / `MAX_ODDS` | 1.5 / 4.5 | Диапазон коэффициентов |
| `COMPETITIONS` | PL,PD,SA,BL1,FL1,CL | Какие турниры тянуть |

## Что улучшить дальше

- Подключить odds-провайдер (the-odds-api, API-Football) для реальных value bets.
- Добавить xG / xA фичи с понимающего их источника.
- Ансамбль из нескольких моделей (LightGBM, нейросеть) с blending'ом.
- Bankroll management по портфелю — учитывать корреляцию рынков.
