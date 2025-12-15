-- Пример использования расширения
CREATE TABLE user_item (
    user_id TEXT,
    item_id TEXT,
    interaction_time TIMESTAMP DEFAULT NOW()
);

-- Генерация тестовых данных
INSERT INTO user_item (user_id, item_id) VALUES
('user1', 'itemA'),
('user1', 'itemB'),
('user2', 'itemA'),
('user2', 'itemC'),
('user3', 'itemB'),
('user3', 'itemD');

-- Создание и обучение модели
SELECT recsys.create_new_model() as model_id;

SELECT recsys.train('user_item', 'user_id', 'item_id', 1);

-- Проверка статуса
SELECT * FROM recsys.trained_models;

-- Получение рекомендаций
SELECT * FROM recsys.user_item_recommend(
    1, 'user2', 'user_item', 'user_id', 'item_id', 5
);

-- Item-to-item рекомендации
SELECT * FROM recsys.item_item_recommend(1, 'itemA', 5);