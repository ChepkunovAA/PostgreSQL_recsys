#include "manager.h"

#ifdef Min
#undef Min
#endif

#ifdef Max
#undef Max
#endif

#ifdef LOG
#undef LOG
#endif

extern "C" {
    #include <postgres.h>
    #include <executor/spi.h>
    #include <fmgr.h>
    #include <funcapi.h>
    #include <miscadmin.h>
    #include <utils/builtins.h>
    #include <utils/tuplestore.h>

    #ifdef PG_MODULE_MAGIC
    PG_MODULE_MAGIC;
    #endif

    // Вспомогательная функция: получить все последовательности для обучения
    torch::Tensor* get_training_data(
        const char* dataset_name,
        const char* user_column,
        const char* item_column,
        const char* order_column,
        int max_seq_len,
        int64_t pad_token = 0
    ) {
        
        StringInfoData query;
        initStringInfo(&query);
        appendStringInfo(&query,
            "SELECT %s, %s, %s FROM %s ORDER BY %s",
            user_column, item_column, order_column, dataset_name, order_column
        );
        
        int ret = SPI_execute(query.data, true, 0);
        if (ret != SPI_OK_SELECT) {
            SPI_finish();
            pfree(query.data);
            elog(ERROR, "Failed to fetch training data");
        }
        
        size_t total_rows = SPI_processed;
        SPITupleTable* tuptable = SPI_tuptable;
        
        // Группируем взаимодействия по пользователям
        std::unordered_map<int, std::vector<int>> user_sequences;
        
        for (size_t i = 0; i < total_rows; i++) {
            HeapTuple tuple = tuptable->vals[i];
            char* user_id_str = SPI_getvalue(tuple, tuptable->tupdesc, 1);
            char* item_id_str = SPI_getvalue(tuple, tuptable->tupdesc, 2);
            
            int user_id = std::stoi(user_id_str);
            int item_id = std::stoi(item_id_str);
            user_sequences[user_id].push_back(item_id);
            
            pfree(user_id_str);
            pfree(item_id_str);
        }
        
        // Создаем батч последовательностей
        std::vector<torch::Tensor> sequences;
        
        for (auto& [user_id, items] : user_sequences) {
            if (items.size() < 2) continue;  // Нужно минимум 2 элемента
            
            // Создаем несколько обучающих примеров из одной последовательности
            for (size_t i = 1; i < items.size(); i++) {
                auto options = torch::TensorOptions().dtype(torch::kInt64);
                torch::Tensor seq = torch::full({1, max_seq_len + 1}, pad_token, options);
                
                // Заполняем последовательность
                int start_pos = std::max(0, (int)i + 1 - max_seq_len - 1);
                for (size_t j = start_pos; j <= i; j++) {
                    int pos = max_seq_len + 1 - (i - j + 1);
                    seq[0][pos] = items[j];
                }
                
                sequences.push_back(seq);
            }
        }
        
        pfree(query.data);
        
        // Объединяем все последовательности в один тензор
        if (sequences.empty()) {
            elog(ERROR, "No valid training sequences found");
        }
        torch::Tensor* result = new torch::Tensor(torch::cat(sequences, 0));
        return result;
    }

        // Вспомогательная функция: получить последовательность пользователя
    torch::Tensor* get_user_sequence(
        const char* dataset_name,
        const char* user_column,
        const char* item_column,
        const char* order_column,
        int user_id,
        int max_seq_len,
        int64_t pad_token = 0
    ) {
        if (SPI_connect() != SPI_OK_CONNECT) {
            elog(ERROR, "SPI_connect failed in get_user_sequence");
        }
        
        // Получаем последние max_seq_len взаимодействий пользователя
        StringInfoData query;
        initStringInfo(&query);
        appendStringInfo(&query,
            "SELECT %s, %s FROM %s WHERE %s = '%d' ORDER BY %s DESC LIMIT %d",
            item_column, order_column, dataset_name, user_column, user_id, order_column, max_seq_len
        );
        
        int ret = SPI_execute(query.data, true, 0);
        if (ret != SPI_OK_SELECT) {
            SPI_finish();
            pfree(query.data);
            elog(ERROR, "Failed to fetch user sequence");
        }
        
        size_t num_items = SPI_processed;
        
        // Создаем тензор с паддингом
        auto options = torch::TensorOptions().dtype(torch::kInt64);
        torch::Tensor sequence = torch::full({1, max_seq_len}, pad_token, options);
        
        // Заполняем с конца (более старые взаимодействия слева)
        SPITupleTable* tuptable = SPI_tuptable;
        for (size_t i = 0; i < num_items; i++) {
            HeapTuple tuple = tuptable->vals[i];
            char* item_id_str = SPI_getvalue(tuple, tuptable->tupdesc, 1);
            int64_t item_id = std::stoll(item_id_str);
            
            // Размещаем от старых к новым
            int pos = max_seq_len - num_items + i;
            sequence[0][pos] = item_id;
            
            pfree(item_id_str);
        }

        pfree(query.data);
        SPI_finish();

        torch::Tensor* result = new torch::Tensor(sequence);
        return result;
    }

    void _PG_init(void) {
        // Инициализируем менеджер моделей
        get_model_manager_instance();
    }

    void _PG_fini(void) {
        get_model_manager_instance()->save_all();
    }

    PG_FUNCTION_INFO_V1(train_internal);

    Datum train_internal(PG_FUNCTION_ARGS) {
        // Получаем аргументы
        text* dataset = PG_GETARG_TEXT_PP(0);
        text* user_col = PG_GETARG_TEXT_PP(1);
        text* item_col = PG_GETARG_TEXT_PP(2);
        text* order_col = PG_GETARG_TEXT_PP(3);
        int model_id = PG_GETARG_INT32(4);

        char* dataset_str = text_to_cstring(dataset);
        char* user_str = text_to_cstring(user_col);
        char* item_str = text_to_cstring(item_col);
        char* order_str = text_to_cstring(order_col);

        int ret;
        StringInfoData query;
        
        if (SPI_connect() != SPI_OK_CONNECT) {
            elog(ERROR, "SPI_connect failed");
        }

        // Получаем число уникальных 
        initStringInfo(&query);
        appendStringInfo(&query, 
                        "SELECT MAX(%s) FROM %s",
                        item_str, dataset_str);
        
        ret = SPI_execute(query.data, true, 0);
        if (ret != SPI_OK_SELECT) {
            elog(ERROR, "Ошибка при получении уникальных items");
        }
        SPITupleTable* tuptable = SPI_tuptable;
        HeapTuple tuple = tuptable->vals[0];
        char* num_items_str = SPI_getvalue(tuple, tuptable->tupdesc, 1);
        int64_t num_items = std::stoll(num_items_str);
        pfree(num_items_str);

        auto manager = get_model_manager_instance();

        bool created = manager->create_model(
            model_id,
            num_items,
            100,
            128,
            2,
            4,
            512,
            0.1
        );
        if (!created) {
            elog(ERROR, "Не удалось создать модель %d", model_id);
        }
        torch::Tensor* train_data = get_training_data(
            dataset_str,
            user_str,
            item_str,
            order_str,
            100
        );

        // Обучаем модель
        manager->train_model(
            model_id,
            *train_data,
            10,
            100,
            0.1
        );

        delete train_data;

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

        auto emb_tensor = manager->get_embeddings(model_id);
        auto item_embeddings = emb_tensor.accessor<float, 2>();

        // Инициализируем эмбеддинги
        for (int i = 0; i < num_items; i++) {
            initStringInfo(&query);
            appendStringInfo(&query, 
                            "INSERT INTO recsys.item_embeddings (model_id, item_id, embedding) "
                            "VALUES (%d, '%d', '[", 
                            model_id, i+1);

            for (int j = 0; j < 128; j++) {
                float emb_val = item_embeddings[i][j];
                appendStringInfo(&query, "%f", emb_val);
                
                if (j < 127) {
                    appendStringInfoString(&query, ", ");
                }
            }
            
            appendStringInfoString(&query, "]')");
            
            ret = SPI_execute(query.data, false, 0);
            if (ret != SPI_OK_INSERT) {
                elog(ERROR, "Ошибка при вставке эмбеддинга для товара %d", i+1);
            }
        }
        
        pfree(query.data);
        
        // Освобождаем ресурсы
        SPI_finish();
        
        pfree(dataset_str);
        pfree(user_str);
        pfree(item_str);
        pfree(order_str); 
        
        PG_RETURN_VOID();
    }

    PG_FUNCTION_INFO_V1(recommend_internal);

    Datum recommend_internal(PG_FUNCTION_ARGS) {
        ReturnSetInfo* rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;

        if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
            ereport(ERROR, (errcode(ERRCODE_FEATURE_NOT_SUPPORTED), errmsg("set-valued function called in context that cannot accept a set")));
        if (!(rsinfo->allowedModes & SFRM_Materialize))
            ereport(ERROR, (errcode(ERRCODE_SYNTAX_ERROR), errmsg("materialize mode required, but it is not allowed in this context")));


        // Получаем аргументы
        int model_id = PG_GETARG_INT32(0);
        int user_id = PG_GETARG_INT32(1);
        text* dataset_name = PG_GETARG_TEXT_PP(2);
        text* user_column = PG_GETARG_TEXT_PP(3);
        text* item_column = PG_GETARG_TEXT_PP(4);
        text* order_column = PG_GETARG_TEXT_PP(5);
        int32 top_k = PG_GETARG_INT32(6);
        float8 min_score = PG_GETARG_FLOAT8(7);
        
        char* dataset_name_str = text_to_cstring(dataset_name);
        char* user_column_str = text_to_cstring(user_column);
        char* item_column_str = text_to_cstring(item_column);
        char* order_column_str = text_to_cstring(order_column);

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

        auto manager = get_model_manager_instance();
    
        // Загружаем модель, если не загружена
        if (!manager->has_model(model_id)) {
            bool loaded = manager->load_model(model_id);
            if (!loaded) {
                elog(ERROR, "Модель %d не найдена", model_id);
            }
        }
        
        torch::Tensor* user_history = get_user_sequence(
            dataset_name_str,
            user_column_str,
            item_column_str,
            order_column_str,
            user_id,
            100
        );

        TopKResult predictions = manager->predict(
            model_id,
            *user_history,
            top_k
        );
        auto top_indices = predictions.indices.to(torch::kCPU).detach();
        auto top_scores = predictions.values.to(torch::kCPU).detach();
        delete user_history;

        // Конвертируем тензор в результаты PostgreSQL
        for (int k = 0; k < top_k; k++) {
            float score = top_scores[k].item<float>();
            int item_idx = top_indices[k].item<int>();
            
            if (item_idx == 0) continue;
            if (score < min_score) continue;
            
            char item_id_str[32];
            snprintf(item_id_str, sizeof(item_id_str), "%d", item_idx);
            
            Datum values[2];
            bool nulls[2] = {false, false};
            values[0] = CStringGetTextDatum(item_id_str);
            values[1] = Float8GetDatum(score);
            
            HeapTuple result_tuple = heap_form_tuple(tupdesc, values, nulls);
            tuplestore_puttuple(tupstore, result_tuple);
            heap_freetuple(result_tuple);
        }

        pfree(dataset_name_str);
        pfree(user_column_str);
        pfree(item_column_str);
        pfree(order_column_str);

        return (Datum)0;
    }

    PG_FUNCTION_INFO_V1(is_model_loaded_internal);
    Datum is_model_loaded_internal(PG_FUNCTION_ARGS)
    {
        int model_id = PG_GETARG_INT32(0);
        auto manager = get_model_manager_instance();
        bool exists = manager->has_model(model_id);
        PG_RETURN_BOOL(exists);
    }

    PG_FUNCTION_INFO_V1(load_model_internal);
    Datum load_model_internal(PG_FUNCTION_ARGS)
    {
        int model_id = PG_GETARG_INT32(0);
        auto manager = get_model_manager_instance();
        bool exists = manager->load_model(model_id);
        PG_RETURN_BOOL(exists);
    }

    PG_FUNCTION_INFO_V1(unload_model_internal);
    Datum unload_model_internal(PG_FUNCTION_ARGS)
    {
        int model_id = PG_GETARG_INT32(0);
        auto manager = get_model_manager_instance();
        bool exists = manager->unload_model(model_id);
        PG_RETURN_BOOL(exists);
    }

    PG_FUNCTION_INFO_V1(save_model_internal);
    Datum save_model_internal(PG_FUNCTION_ARGS)
    {
        int model_id = PG_GETARG_INT32(0);
        auto manager = get_model_manager_instance();
        bool exists = manager->save_model(model_id);
        PG_RETURN_BOOL(exists);
    }
    
    PG_FUNCTION_INFO_V1(delete_model_internal);
    Datum delete_model_internal(PG_FUNCTION_ARGS)
    {
        int model_id = PG_GETARG_INT32(0);
        auto manager = get_model_manager_instance();
        bool exists = manager->delete_model(model_id);
        PG_RETURN_BOOL(exists);
    }
}