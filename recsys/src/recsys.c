#include <postgres.h>
#include <executor/spi.h>
#include <fmgr.h>
#include <funcapi.h>
#include <miscadmin.h>
#include <utils/builtins.h>
#include <utils/tuplestore.h>

#include "model.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(train_internal);

Datum train_internal(PG_FUNCTION_ARGS) {
    // Получаем аргументы
    text* dataset = PG_GETARG_TEXT_PP(0);
    text* user_col = PG_GETARG_TEXT_PP(1);
    text* item_col = PG_GETARG_TEXT_PP(2);
    int32 model_id = PG_GETARG_INT32(3);

    char* dataset_str = text_to_cstring(dataset);
    char* user_str = text_to_cstring(user_col);
    char* item_str = text_to_cstring(item_col);

    int ret;
    StringInfoData query;
    
    if (SPI_connect() != SPI_OK_CONNECT) {
        elog(ERROR, "SPI_connect failed");
    }

    // Получаем список уникальных 
    initStringInfo(&query);
    appendStringInfo(&query, 
                     "SELECT DISTINCT %s FROM %s",
                     item_str, dataset_str);
    
    ret = SPI_execute(query.data, true, 0);
    if (ret != SPI_OK_SELECT) {
        elog(ERROR, "Ошибка при получении уникальных items");
    }
    uint32 distinct_count = SPI_processed;

    // Инициализируем эмбеддинги случайными векторами
    SPITupleTable* tuptable = SPI_tuptable;
    for (uint32 i = 0; i < distinct_count; i++) {
        HeapTuple tuple = tuptable->vals[i];
        TupleDesc tupdesc = tuptable->tupdesc;

        char* item_id_str = SPI_getvalue(tuple, tupdesc, 1);
        initStringInfo(&query);
        appendStringInfo(&query, 
                         "INSERT INTO recsys.item_embeddings (model_id, item_id, embedding) "
                         "VALUES (%d, %s, '[", 
                         model_id, item_id_str);

        for (int j = 0; j < 128; j++) {
            double random_val = 2.0 * ((double) random() / (double) RAND_MAX) - 1.0;
            appendStringInfo(&query, "%f", random_val);
            
            if (j < 127) {
                appendStringInfoString(&query, ", ");
            }
        }
        
        appendStringInfoString(&query, "]')");
        
        ret = SPI_execute(query.data, false, 0);
        if (ret != SPI_OK_INSERT) {
            elog(ERROR, "Ошибка при вставке эмбеддинга для товара %s", item_id_str);
        }
        pfree(item_id_str);
        pfree(query.data);
        
    }

    // Обновляем статус модели
    initStringInfo(&query);
    appendStringInfo(&query, 
                     "UPDATE recsys.models SET model_status = 'ready', "
                     "updated_at = CURRENT_TIMESTAMP "
                     "WHERE model_id = %d", 
                     model_id);
    
    ret = SPI_execute(query.data, false, 0);
    if (ret != SPI_OK_UPDATE) {
        elog(ERROR, "Ошибка при обновлении статуса модели %d", model_id);
    }
    
    pfree(query.data);
    
    // Освобождаем ресурсы
    SPI_finish();
    
    pfree(dataset_str);
    pfree(user_str);
    pfree(item_str);
    
    PG_RETURN_VOID();
}

PG_FUNCTION_INFO_V1(recommend_internal);

Datum recommend_internal(PG_FUNCTION_ARGS) {
    ReturnSetInfo* rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;

    /* check to see if caller supports us returning a tuplestore */
    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("set-valued function called in context that cannot accept a set")));
    if (!(rsinfo->allowedModes & SFRM_Materialize))
        ereport(ERROR, (errcode(ERRCODE_SYNTAX_ERROR), errmsg("materialize mode required, but it is not allowed in this context")));


    // Получаем аргументы
    int32 model_id = PG_GETARG_INT32(0);
    text* user_id = PG_GETARG_TEXT_PP(1);
    text* dataset_name = PG_GETARG_TEXT_PP(2);
    text* user_column = PG_GETARG_TEXT_PP(3);
    text* item_column = PG_GETARG_TEXT_PP(4);
    int32 top_k = PG_GETARG_INT32(5);
    float8 min_score = PG_GETARG_FLOAT8(6);
    
    char* dataset_name_str = text_to_cstring(dataset_name);
    char* user_column_str = text_to_cstring(user_column);
    char* item_column_str = text_to_cstring(item_column);
    char* user_id_str = text_to_cstring(user_id);

    /* Настраиваем вывод TupleDesc */
    MemoryContext oldcontext = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    TupleDesc tupdesc = CreateTemplateTupleDesc(2);
    TupleDescInitEntry(tupdesc, (AttrNumber)1, "recommended_item_id", TEXTOID, -1, 0);
    TupleDescInitEntry(tupdesc, (AttrNumber)2, "score", FLOAT8OID, -1, 0);
    tupdesc = BlessTupleDesc(tupdesc);
    bool randomAccess = (rsinfo->allowedModes & SFRM_Materialize_Random) != 0;
    Tuplestorestate* tupstore = tuplestore_begin_heap(randomAccess, false, work_mem);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = tupstore;
    rsinfo->setDesc = tupdesc;
    MemoryContextSwitchTo(oldcontext);

    // Здесь должна быть основная логика рекомендаций:
    // 1. Загружаем обученную модель по model_id
    // 2. Получаем эмбеддинг пользователя
    // 3. Для каждого айтема вычисляем score и similarity
    // 4. Сортируем по score и возвращаем топ-k результатов
    // Временно возвращаем случайный результат
    if (SPI_connect() == SPI_OK_CONNECT) {
        StringInfoData query;
        initStringInfo(&query);
        
        appendStringInfo(&query, 
            "SELECT DISTINCT %s FROM %s LIMIT %d",
            item_column_str,
            dataset_name_str,
            top_k);
        
        int ret = SPI_execute(query.data, true, 0);
        
        if (ret == SPI_OK_SELECT) {
            SPITupleTable *tuptable = SPI_tuptable;
            uint32 ntuples = SPI_processed;
            
            for (uint32 i = 0; i < ntuples; i++) {
                HeapTuple tuple = tuptable->vals[i];
                char *item_id = SPI_getvalue(tuple, tuptable->tupdesc, 1);
                
                double score = ((double)random() / RAND_MAX);
                
                Datum values[2];
                bool nulls[2] = {false, false};
                values[0] = CStringGetTextDatum(item_id);
                values[1] = Float8GetDatum(score);
                
                HeapTuple result_tuple = heap_form_tuple(tupdesc, values, nulls);
                tuplestore_puttuple(tupstore, result_tuple);
                
                pfree(item_id);
                heap_freetuple(result_tuple);
            }
            SPI_freetuptable(tuptable);
        }
        
        pfree(query.data);
        SPI_finish();
    }
    
    tuplestore_donestoring(tupstore);

    pfree(dataset_name_str);
    pfree(user_column_str);
    pfree(item_column_str);
    pfree(user_id_str);

    return (Datum)0;
}