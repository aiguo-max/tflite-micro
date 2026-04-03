SLEEP_QEMU_TEST_SRCS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/sleep_test/sleep_qemu_test.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/bidirectional_sequence_gru.cc \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/kernels/layer_norm.cc

SLEEP_QEMU_TEST_HDRS := \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/sleep_test/sleep_model_data.h \
$(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/sleep_test/sleep_test_data.h

$(eval $(call microlite_test,sleep_qemu_test,\
$(SLEEP_QEMU_TEST_SRCS),\
$(SLEEP_QEMU_TEST_HDRS),))
