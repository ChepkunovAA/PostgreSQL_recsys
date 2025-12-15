# Разработка расширения для СУБД PostgreSQL, реализующего на уровне СУБД механизм рекомендаций на основе трансформерных моделей (SQL as a recommendation)

### Установка:
1. Build and Install:
```
cd /tmp
git clone https://github.com/ChepkunovAA/PostgreSQL_recsys
cd PostgreSQL_recsys/recsys
make
make install
```
2. Enable the Extension:
```
CREATE EXTENSION pg_background;
```

### Основные архитектурные компоненты:
1. Система хранения моделей
    - Таблица recsys.models с отслеживанием статусов (untrained, training, ready, failed)
    - Прототип типа ModelConfig для хранения параметров моделей

2. Хранение эмбеддингов
    - Таблица recsys.item_embeddings с векторами размерностью 128
    - HNSW-индексы для быстрого поиска схожих векторов
    - Использование расширения vector для векторных операций

3. API для пользователя
    - ```recsys.create_new_model()``` - создание новой модели
    - ```recsys.train()``` - запуск обучения на пользовательских данных
    - ```recsys.user_item_recommend()``` - рекомендации user-to-item
    - ```recsys.item_item_recommend()``` - рекомендации item-to-item

4. Мониторинг и диагностика
    - Представление со статистикой по моделям (статус, время создания и обновления,количество созданных эмбеддингов)
    - Представления по статусам моделей (untrained_models, training_models, и т.д.)
    - Функция проверки существования и статуса модели

5. Реализованные API
    - Cоздание и управление жизненным циклом (create-train-recommend) моделей
    - Хранение векторных представлений (эмбеддингов) товаров
    - Два типа рекомендаций: user-item и item-item
    - Проверка корректности входных данных (таблиц, столбцов, статусов моделей)

### Демонстрационные материалы:
Пример использования основных API можно найти в examples/quickexample.sql
