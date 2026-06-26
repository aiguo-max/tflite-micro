$(eval $(call microlite_test,measure_arena,\
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/measure_arena/measure_arena.cc,\
    $(TENSORFLOW_ROOT)tensorflow/lite/micro/examples/measure_arena))
