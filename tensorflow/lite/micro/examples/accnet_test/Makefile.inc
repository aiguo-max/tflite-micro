ACCNET_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_test.cc

ACCNET_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_int8_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_int8_test_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_skip_stage0_conv_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_skip_stage0_conv_test_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_skip_s0_conv1_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_test/accnet_skip_s0_conv1_test_data.h

$(eval $(call microlite_test,accnet_test,\
$(ACCNET_TEST_SRCS),\
$(ACCNET_TEST_HDRS),))
