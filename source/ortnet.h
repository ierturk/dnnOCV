#pragma once
#pragma once
#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

//*****************************************************************************
// helper function to check for status
#define CHECK_STATUS(expr)                               \
  {                                                      \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      exit(1);                                           \
    }                                                    \
  }