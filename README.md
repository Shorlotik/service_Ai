# ML API Wrapper Service

Сервис-обертка для ML API, который предоставляет единообразный интерфейс для классификации текста и анализа тональности. Сервис инкапсулирует взаимодействие с различными ML моделями (локальными в Docker или публичными API), добавляет кеширование, обработку ошибок и логирование.

## Возможности

- 🔌 Поддержка нескольких ML провайдеров:
  - Локальная модель в Docker-контейнере
  - Hugging Face Inference API
  - OpenAI API
- 💾 Кеширование результатов (in-memory или Redis)
- 📊 Метрики использования и производительности
- 🏥 Health check endpoint для мониторинга
- 📝 Структурированное логирование
- 🐳 Docker контейнеризация
- ✅ Unit-тесты с мокированием

## Требования

- Python 3.10+
- Docker и Docker Compose (для запуска через Docker)
- Redis (опционально, если используется Redis кеш)

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd service_Ai
```

### 2. Настройка переменных окружения

Скопируйте `.env.example` в `.env` и настройте переменные:

```bash
cp .env.example .env
```

Отредактируйте `.env` файл и укажите необходимые параметры (см. раздел "Конфигурация" ниже).

### 3. Запуск через Docker Compose (рекомендуется)

```bash
docker-compose up -d
```

Сервис будет доступен по адресу: http://localhost:8000

### 4. Запуск локально (для разработки)

```bash
# Создание виртуального окружения
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск сервиса
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Конфигурация

### Переменные окружения

Основные переменные окружения (полный список см. в `.env.example`):

#### ML Provider Configuration

```bash
# Выбор провайдера: local, huggingface, openai
ML_PROVIDER=local

# Для локальной модели
LOCAL_MODEL_URL=http://localhost:8000/predict

# Для Hugging Face
HUGGINGFACE_API_KEY=your_api_key_here
HUGGINGFACE_MODEL=cardiffnlp/twitter-roberta-base-sentiment-latest

# Для OpenAI
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
```

#### Cache Configuration

```bash
# Стратегия кеширования: memory, redis
CACHE_STRATEGY=redis

# TTL кеша в секундах (по умолчанию 86400 = 24 часа)
CACHE_TTL=86400

# Redis настройки (если используется Redis)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
```

#### Logging Configuration

```bash
# Формат логирования: simple, json
LOG_FORMAT=simple

# Уровень логирования: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

## Получение API ключей

### Hugging Face API Key

1. Зарегистрируйтесь на [Hugging Face](https://huggingface.co/)
2. Перейдите в [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Создайте новый токен с правами чтения
4. Скопируйте токен в переменную `HUGGINGFACE_API_KEY` в `.env` файле

### OpenAI API Key

1. Зарегистрируйтесь на [OpenAI Platform](https://platform.openai.com/)
2. Перейдите в [API Keys](https://platform.openai.com/api-keys)
3. Создайте новый API ключ
4. Скопируйте ключ в переменную `OPENAI_API_KEY` в `.env` файле

**Важно:** Никогда не коммитьте `.env` файл с реальными API ключами в репозиторий!

## API Endpoints

### POST /classify

Классификация текста.

**Запрос:**
```bash
curl -X POST "http://localhost:8000/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

**Ответ (простой формат):**
```json
{
  "label": "POSITIVE",
  "confidence": 0.95,
  "cached": false
}
```

**Ответ (детальный формат):**
```json
{
  "label": "POSITIVE",
  "confidence": 0.95,
  "labels": [
    {"label": "POSITIVE", "confidence": 0.95},
    {"label": "NEGATIVE", "confidence": 0.05}
  ],
  "cached": false
}
```

### GET /health

Проверка работоспособности сервиса.

**Запрос:**
```bash
curl http://localhost:8000/health
```

**Ответ:**
```json
{
  "status": "healthy",
  "details": {
    "redis": "connected",
    "ml_provider": "local (initialized)"
  }
}
```

### GET /metrics

Получение метрик использования сервиса.

**Запрос:**
```bash
curl http://localhost:8000/metrics
```

**Ответ:**
```json
{
  "total_requests": 100,
  "successful_requests": 95,
  "total_errors": 5,
  "errors_by_type": {
    "timeout": 2,
    "api_error": 3
  },
  "cache_hits": 60,
  "cache_misses": 40,
  "cache_hit_rate": 0.6,
  "average_response_time_seconds": 0.234
}
```

### GET /

Информация о сервисе.

**Запрос:**
```bash
curl http://localhost:8000/
```

**Ответ:**
```json
{
  "service": "ML API Wrapper Service",
  "version": "1.0.0",
  "status": "running",
  "ml_provider": "local",
  "cache_strategy": "memory"
}
```

## Примеры использования

### Python

```python
import httpx

# Классификация текста
response = httpx.post(
    "http://localhost:8000/classify",
    json={"text": "This is amazing!"}
)
result = response.json()
print(f"Label: {result['label']}, Confidence: {result['confidence']}")
```

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function classifyText(text) {
  const response = await axios.post('http://localhost:8000/classify', {
    text: text
  });
  return response.data;
}

classifyText('I love this product!')
  .then(result => {
    console.log(`Label: ${result.label}, Confidence: ${result.confidence}`);
  });
```

## Тестирование

Запуск тестов:

```bash
# Все тесты
pytest

# Конкретный файл тестов
pytest tests/test_main.py

# С выводом подробной информации
pytest -v

# С покрытием кода
pytest --cov=. --cov-report=html
```

## Разработка

### Структура проекта

```
.
├── main.py                 # FastAPI приложение
├── config.py               # Конфигурация
├── models.py               # Pydantic модели
├── logger.py               # Настройка логирования
├── metrics.py              # Сбор метрик
├── utils.py                # Утилиты
├── ml_providers/           # ML провайдеры
│   ├── __init__.py
│   ├── base.py
│   ├── local.py
│   ├── huggingface.py
│   ├── openai.py
│   └── normalizers.py
├── cache/                  # Система кеширования
│   ├── __init__.py
│   ├── base.py
│   ├── memory.py
│   └── redis_cache.py
├── tests/                  # Тесты
│   ├── __init__.py
│   ├── test_main.py
│   ├── test_ml_providers.py
│   └── test_cache.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example            # Пример переменных окружения
├── .gitignore              # Git ignore файл
└── README.md
```

### Локальная разработка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Настройте `.env` файл

3. Запустите Redis (если используется):
```bash
docker run -d -p 6379:6379 redis:7-alpine
```

4. Запустите сервис:
```bash
uvicorn main:app --reload
```

## Docker команды

```bash
# Сборка образа
docker-compose build

# Запуск сервисов
docker-compose up -d

# Просмотр логов
docker-compose logs -f app

# Остановка сервисов
docker-compose down

# Остановка с удалением volumes
docker-compose down -v
```

## Обработка ошибок

Сервис возвращает следующие HTTP коды ошибок:

- `400` - Ошибка валидации входных данных
- `502` - Ошибка или таймаут при обращении к ML API
- `500` - Внутренняя ошибка сервиса

Формат ответа с ошибкой:

```json
{
  "error": "timeout",
  "message": "Request to ML API timed out",
  "details": "Request timed out after 30 seconds"
}
```

## Производительность

- Среднее время ответа из кеша: < 500ms
- Среднее время ответа с вызовом ML API: < 5s (зависит от провайдера)
- Эффективность кеша: > 50% для типичных сценариев


