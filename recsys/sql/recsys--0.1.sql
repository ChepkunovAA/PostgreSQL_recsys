CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA recsys;

CREATE TYPE ModelConfig;

CREATE FUNCTION ModelConfig_in(cstring) RETURNS ModelConfig
	AS 'recsys', 'ModelConfig_in' LANGUAGE C IMMUTABLE STRICT;

CREATE FUNCTION ModelConfig_out(ModelConfig) RETURNS cstring
	AS 'recsys', 'ModelConfig_out' LANGUAGE C IMMUTABLE STRICT;

CREATE TYPE ModelConfig (
    INTERNALLENGTH = 256,
    INPUT = ModelConfig_in,
    OUTPUT = ModelConfig_out
);

CREATE TYPE Status AS ENUM ('untrained', 'training', 'ready', 'failed');

-- Таблица для хранения информации о моделях
CREATE TABLE recsys.models (
    model_id SERIAL PRIMARY KEY,
    model_status Status DEFAULT 'untrained',
    model_config ModelConfig DEFAULT NULL,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Таблица для хранения эмбеддингов
CREATE TABLE recsys.item_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES recsys.models ON DELETE CASCADE,
    item_id TEXT NOT NULL,
    embedding vector(128),
    
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT embeddings_model_item UNIQUE (model_id, item_id)
);

CREATE INDEX ON recsys.item_embeddings(model_id);

CREATE INDEX ON recsys.item_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Представления для дигностики
CREATE VIEW recsys.model_stats AS
    SELECT 
        m.model_id,
        m.model_status,
        m.created_at,
        m.updated_at,
        COUNT(e.embedding_id) as embeddings_count
FROM recsys.models m
LEFT JOIN recsys.item_embeddings e ON m.model_id = e.model_id
GROUP BY m.model_id;

CREATE VIEW recsys.untrained_models AS 
    SELECT model_id, created_at, updated_at FROM recsys.model_stats
    WHERE model_status = 'untrained';

CREATE VIEW recsys.training_models AS 
    SELECT model_id, created_at, updated_at FROM recsys.model_stats
    WHERE model_status = 'training';

CREATE VIEW recsys.trained_models AS 
    SELECT model_id, created_at, updated_at FROM recsys.model_stats
    WHERE model_status = 'ready';

CREATE VIEW recsys.failed_models AS 
    SELECT model_id, created_at, updated_at FROM recsys.model_stats
    WHERE model_status = 'failed';

-- Функция проверки существования и статуса модели
CREATE OR REPLACE FUNCTION recsys._check_model(
    pl_model_id INTEGER,
    expected_status Status DEFAULT NULL
)
RETURNS Status
LANGUAGE plpgsql
AS $$
DECLARE
    real_model_status Status;
BEGIN
    -- Проверяем существование модели
    SELECT m.model_status INTO real_model_status
    FROM recsys.models m
    WHERE m.model_id = pl_model_id;
    
    IF NOT FOUND THEN
        RAISE EXCEPTION 'Модель с ID % не найдена', pl_model_id;
    END IF;
    
    -- Проверяем статус модели, если задан ожидаемый статус
    IF expected_status IS NOT NULL AND real_model_status != expected_status THEN
        RAISE EXCEPTION 'Модель должна быть в статусе "%". Текущий статус: "%"', 
            expected_status, real_model_status;
    END IF;
    
    RETURN real_model_status;
END;
$$;

-- Функция проверки существования таблицы
CREATE OR REPLACE FUNCTION recsys._check_table_exists(
    pl_table_name TEXT,
    pl_schema_name TEXT DEFAULT 'public'
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    is_table_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT FROM information_schema.tables t
        WHERE t.table_schema = pl_schema_name 
        AND t.table_name = pl_table_name
    ) INTO is_table_exists;
    
    IF NOT is_table_exists THEN
        RAISE EXCEPTION 'Таблица "%"."%" не найдена', pl_schema_name, pl_table_name;
    END IF;
    
    RETURN TRUE;
END;
$$;

-- Функция проверки существования столбцов в таблице
CREATE OR REPLACE FUNCTION recsys._check_columns_exist(
    pl_table_name TEXT,
    pl_columns TEXT[],
    pl_schema_name TEXT DEFAULT 'public'
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $$
DECLARE
    pl_column_name TEXT;
    is_column_exists BOOLEAN;
BEGIN
    FOREACH pl_column_name IN ARRAY pl_columns
    LOOP
        SELECT EXISTS (
            SELECT FROM information_schema.columns c
            WHERE c.table_schema = pl_schema_name 
            AND c.table_name = pl_table_name 
            AND c.column_name = pl_column_name
        ) INTO is_column_exists;
        
        IF NOT is_column_exists THEN
            RAISE EXCEPTION 'Столбец "%" не найден в таблице "%"."%"', 
                pl_column_name, pl_schema_name, pl_table_name;
        END IF;
    END LOOP;
    
    RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION recsys.create_new_model()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    new_model_id INTEGER;
BEGIN
    INSERT INTO recsys.models DEFAULT VALUES RETURNING model_id INTO new_model_id;
    RETURN new_model_id;
END;
$$;

-- Функция для запуска обучения модели
CREATE OR REPLACE FUNCTION recsys.train(
    dataset_name TEXT,
    user_column TEXT,
    item_column TEXT,
    target_model_id INTEGER
)
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    table_exists BOOLEAN;
    column_exists BOOLEAN;
    model_status_value STATUS;
BEGIN 
    -- Проверяем существование таблицы
    PERFORM recsys._check_table_exists(dataset_name);
    
    -- Проверяем существование столбцов
    PERFORM recsys._check_columns_exist(dataset_name, ARRAY[user_column, item_column]);

    -- Проверяем существование модели и её статус
    model_status_value := recsys._check_model(target_model_id);
       
    -- Проверяем, что модель в корректном статусе для обучения
    IF model_status_value = 'training' THEN
        RAISE EXCEPTION 'Модель уже обучается';
    END IF;
    IF model_status_value = 'ready' THEN
        RAISE NOTICE 'Модель уже обучена';
        DELETE FROM recsys.item_embeddings
        WHERE model_id = target_model_id;
    END IF;
    
    -- Обновляем статус модели на "training"
    UPDATE recsys.models 
    SET model_status = 'training',
        updated_at = CURRENT_TIMESTAMP
    WHERE model_id = target_model_id;
    
    -- Вызываем C-функцию для обучения
    PERFORM recsys.train_internal(dataset_name, user_column, item_column, target_model_id);
        
EXCEPTION 
    WHEN OTHERS THEN
         -- В случае ошибки обновляем статус на "failed" и пробрасываем ошибку
        UPDATE recsys.models 
        SET model_status = 'failed',
            updated_at = CURRENT_TIMESTAMP
        WHERE model_id = target_model_id;
            
        RAISE;
END;
$$;

-- Объявляем функцию, которая будет реализована в C-расширении
CREATE OR REPLACE FUNCTION recsys.train_internal(
    dataset_table TEXT,
    user_column TEXT,
    item_column TEXT,
    model_id INTEGER
)
RETURNS VOID
LANGUAGE C
STRICT
AS 'recsys', 'train_internal';

-- Функция для получения user-to-item рекомендаций с использованием обученной модели
CREATE OR REPLACE FUNCTION recsys.user_item_recommend(
    target_model_id INTEGER,
    target_user_id TEXT,
    dataset_name TEXT,
    user_column TEXT DEFAULT 'user_id',
    item_column TEXT DEFAULT 'item_id',
    top_k INTEGER DEFAULT 10,
    min_score FLOAT DEFAULT 0.0
)
RETURNS TABLE(
    recommended_item_id TEXT,
    score FLOAT
) 
LANGUAGE plpgsql
AS $$
BEGIN
    -- Проверяем существование таблицы
    PERFORM recsys._check_table_exists(dataset_name);
    
    -- Проверяем существование столбцов
    PERFORM recsys._check_columns_exist(dataset_name, ARRAY[user_column, item_column]);

    -- Проверяем существование модели и её статус
    PERFORM recsys._check_model(target_model_id, 'ready');

    -- Возвращаем рекомендации, вызывая внутреннюю C-функцию
    RETURN QUERY SELECT * FROM recsys.recommend_internal(
        target_model_id,
        target_user_id,
        dataset_name,
        user_column,
        item_column,
        top_k,
        min_score
    );  
END;
$$;

-- Объявляем внутреннюю C-функцию для генерации рекомендаций
CREATE OR REPLACE FUNCTION recsys.recommend_internal(
    model_id INTEGER,
    user_id TEXT,
    dataset_name TEXT,
    user_column TEXT,
    item_column TEXT,
    top_k INTEGER,
    min_score FLOAT
)
RETURNS TABLE(
    recommended_item_id TEXT,
    score FLOAT
)
LANGUAGE C
AS 'recsys', 'recommend_internal';

-- Функция для получения item-to-item рекомендаций с использованием обученной модели
CREATE OR REPLACE FUNCTION recsys.item_item_recommend(
    target_model_id INTEGER,
    target_item_id TEXT,
    top_k INTEGER DEFAULT 10,
    min_similarity FLOAT DEFAULT 0.0
)
RETURNS TABLE(
    recommended_item_id TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    -- Проверяем, что модель существует и готова
    PERFORM recsys._check_model(target_model_id, 'ready');
    
    -- Проверяем, что для данного item_id есть эмбеддинг
    IF NOT EXISTS (
        SELECT 1 FROM recsys.item_embeddings 
        WHERE model_id = target_model_id AND item_id = target_item_id
    ) THEN
        RAISE EXCEPTION 'Эмбеддинг для item_id % не найден в модели %', target_item_id, target_model_id;
    END IF;
    
    RETURN QUERY
    WITH filtred_embeddings AS 
        (SELECT * FROM recsys.item_embeddings
        WHERE model_id = target_model_id)
    SELECT 
        e2.item_id AS recommended_item_id,
        1 - (e1.embedding <=> e2.embedding) AS similarity
    FROM filtred_embeddings e1
    CROSS JOIN filtred_embeddings e2
    WHERE e1.item_id = target_item_id
        AND e2.item_id != target_item_id
        AND similarity > min_similarity
    ORDER BY e1.embedding <=> e2.embedding
    LIMIT top_k;
END;
$$;