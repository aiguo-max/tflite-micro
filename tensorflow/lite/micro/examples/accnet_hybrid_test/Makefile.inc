ACCNET_HYBRID_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_hybrid_test/accnet_hybrid_test.cc

ACCNET_HYBRID_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_hybrid_test/accnet_skip_stage0_conv_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_hybrid_test/accnet_skip_stage0_conv_test_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_hybrid_test/accnet_skip_s0_conv1_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/accnet_hybrid_test/accnet_skip_s0_conv1_test_data.h

$(eval $(call microlite_test,accnet_hybrid_test,\
$(ACCNET_HYBRID_TEST_SRCS),\
$(ACCNET_HYBRID_TEST_HDRS),))
