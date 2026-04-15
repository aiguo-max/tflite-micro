HR_TFLM_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hr_test/hr_tflm_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/layer_norm.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/unidirectional_sequence_gru.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/multi_head_attention.cc

HR_TFLM_TEST_HDRS :=

$(eval $(call microlite_test,hr_tflm_test,\
$(HR_TFLM_TEST_SRCS),\
$(HR_TFLM_TEST_HDRS),))
