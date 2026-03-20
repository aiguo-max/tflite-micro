HYBRID_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hybrid_test/hybrid_test.cc

HYBRID_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hybrid_test/skip_s0_conv1_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hybrid_test/skip_s0_conv1_test_data.h

$(eval $(call microlite_test,hybrid_test,\
$(HYBRID_TEST_SRCS),\
$(HYBRID_TEST_HDRS),))
