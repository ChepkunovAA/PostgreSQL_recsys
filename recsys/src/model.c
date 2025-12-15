#include "model.h"
#include <string.h>

PG_FUNCTION_INFO_V1(ModelConfig_in);

Datum ModelConfig_in(PG_FUNCTION_ARGS)
{
	char* str = PG_GETARG_CSTRING(0);

	ModelConfig* mc = (ModelConfig*)palloc(sizeof(ModelConfig));

	strncpy(mc->path_to_weights, str, 255);

	PG_RETURN_POINTER(mc);
}

PG_FUNCTION_INFO_V1(ModelConfig_out);

Datum ModelConfig_out(PG_FUNCTION_ARGS)
{
	ModelConfig*  mc = (ModelConfig*)PG_GETARG_POINTER(0);

	char* str = (char*)palloc(256);

	snprintf(str, 255, "%s", mc->path_to_weights);

	PG_RETURN_CSTRING(str);
}