ACCNET_INT8_QEMU_TEST_SRCS := tensorflow/lite/micro/examples/accnet_test/accnet_int8_qemu_test.cc
ACCNET_INT8_QEMU_TEST_HDRS := tensorflow/lite/micro/examples/accnet_test/accnet_int8_model_data.h \
  tensorflow/lite/micro/examples/accnet_test/accnet_int8_test_data.h
$(eval $(call microlite_test,accnet_int8_qemu_test,$(ACCNET_INT8_QEMU_TEST_SRCS),$(ACCNET_INT8_QEMU_TEST_HDRS)))
