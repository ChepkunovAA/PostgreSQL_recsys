DROP EXTENSION IF EXISTS recsys;
CREATE EXTENSION IF NOT EXISTS recsys CASCADE;

DROP TABLE IF EXISTS ratings;
CREATE TABLE ratings (
    user_id INTEGER,
    item_id INTEGER,
    rating INTEGER,
    interaction_id INTEGER
);

CREATE TEMP TABLE temp_import (LIKE your_table INCLUDING DEFAULTS);
ALTER TABLE temp_import DROP COLUMN interaction_id;

COPY temp_import(user_id, item_id, rating)
FROM 'ratings.dat'
DELIMITER ';';

INSERT INTO ratings (user_id, item_id, rating, interaction_id)
SELECT 
    user_id, 
    item_id, 
    rating, 
    ROW_NUMBER() OVER ()
FROM temp_import;

SELECT recsys.create_new_model() as model_id;

SELECT recsys.train('ratings', 'user_id', 'item_id', 'interaction_id', 1);

SELECT * FROM recsys.user_item_recommend(
    1, 2, 'user_item', 'user_id', 'item_id', 'interaction_time', 10
);