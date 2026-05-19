-- Проверка наличия модели в оперативной памяти
CREATE OR REPLACE FUNCTION recsys.is_model_loaded(
    model_id INTEGER
)
RETURNS BOOLEAN
LANGUAGE C
STRICT
AS 'recsys', 'is_model_loaded_internal';

-- Загрузка модели по ID в оперативную память
CREATE OR REPLACE FUNCTION recsys.load_model(
    model_id INTEGER
)
RETURNS VOID
LANGUAGE C
STRICT
AS 'recsys', 'load_model_internal';

-- Выгрузка модели по ID из оперативной памяти
CREATE OR REPLACE FUNCTION recsys.unload_model(
    model_id INTEGER
)
RETURNS VOID
LANGUAGE C
STRICT
AS 'recsys', 'unload_model_internal';

-- Сохранение модели на диск
CREATE OR REPLACE FUNCTION recsys.save_model(
    model_id INTEGER
)
RETURNS VOID
LANGUAGE C
STRICT
AS 'recsys', 'save_model_internal';

-- Удаление модели по ID
CREATE OR REPLACE FUNCTION recsys.delete_model(
    model_id INTEGER
)
RETURNS VOID
LANGUAGE C
STRICT
AS 'recsys', 'delete_model_internal';