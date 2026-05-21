DROP EXTENSION IF EXISTS recsys;
CREATE EXTENSION IF NOT EXISTS recsys;

-- Пример использования расширения
DROP TABLE IF EXISTS user_item;
CREATE TABLE user_item (
    user_id INTEGER,
    item_id INTEGER,
    interaction_time TIMESTAMP DEFAULT NOW()
);

-- Генерация тестовых данных
INSERT INTO user_item (user_id, item_id) VALUES
(1, 1),
(1, 2),
(2, 1),
(2, 3),
(2, 2),
(3, 4);

-- Создание и обучение модели
SELECT recsys.create_new_model() as model_id;

SELECT recsys.train('user_item', 'user_id', 'item_id', 'interaction_time', 1);

-- Проверка статуса
SELECT * FROM recsys.trained_models;

-- Получение рекомендаций
SELECT * FROM recsys.user_item_recommend(
    1, 2, 'user_item', 'user_id', 'item_id', 'interaction_time', 3
);

-- Item-to-item рекомендации
SELECT * FROM recsys.item_item_recommend(1, 1, 2);