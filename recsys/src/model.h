#include <postgres.h>
#include <fmgr.h>
#include <libpq/pqformat.h>
#include <math.h>

typedef struct ModelConfig
{
        /* Структура для хранения параметров модели, в будущем будет дополнена */
        char path_to_weights[256];
   
} ModelConfig;
