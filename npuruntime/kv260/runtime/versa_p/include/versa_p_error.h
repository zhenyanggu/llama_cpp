#ifndef VERSA_P_ERROR_H
#define VERSA_P_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum versa_p_status {
    VERSA_P_OK = 0,
    VERSA_P_ERR_INVAL = -1,
    VERSA_P_ERR_IO = -2,
    VERSA_P_ERR_NOMEM = -3,
    VERSA_P_ERR_TIMEOUT = -4,
    VERSA_P_ERR_BUSY = -5,
    VERSA_P_ERR_ALIGNMENT = -6,
    VERSA_P_ERR_ILLEGAL_SHAPE = -7,
    VERSA_P_ERR_RESOURCE_CONFLICT = -8,
    VERSA_P_ERR_BANK_CONFLICT = -9,
    VERSA_P_ERR_BANK_NOT_VALID = -10,
    VERSA_P_ERR_SHAPE_MISMATCH = -11,
    VERSA_P_ERR_UNSUPPORTED_MODE = -12,
    VERSA_P_ERR_ILLEGAL_FLAGS = -13,
    VERSA_P_ERR_HARDWARE = -14
} versa_p_status;

const char *versa_p_status_string(int status);

#ifdef __cplusplus
}
#endif

#endif
