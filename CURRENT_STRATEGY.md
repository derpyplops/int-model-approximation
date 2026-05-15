# Current Strategy

Compared with the baseline integerized model, FP8 checkpoint linears run two Freivalds-checkable integer GEMMs: the primary high-precision int32 product and a second product over exact FP8-codebook integer values; the output uses deterministic dyadic postprocessing, `Y = Y_high + 10/32 * (Y_codebook - Y_high)`, while non-FP8 linears keep the baseline int32 GEMM path.
