HR_PNET_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hr_pnet_test/hr_pnet_test.cc

HR_PNET_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hr_pnet_test/pnet_v2_skip_s0_conv1_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/hr_pnet_test/pnet_v2_skip_s0_conv1_test_data.h

$(eval $(call microlite_test,hr_pnet_test,\
$(HR_PNET_TEST_SRCS),\
$(HR_PNET_TEST_HDRS),))
