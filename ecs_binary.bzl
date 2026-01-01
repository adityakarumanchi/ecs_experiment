load("@rules_cc//cc:defs.bzl", "cc_binary")

def ecs_cc_binary(name, srcs):
    cc_binary(
        name = name,
        deps = ["@eigen",
                "@google_benchmark//:benchmark",
            ],
        srcs = srcs,
        # data = [],
        additional_linker_inputs = [],
        # args = [""],
        # copts = ["-std=c++17", "-O3"],
        # includes = [""],
        # linkopts = ["-lbenchmark -lpthread"],
        linkshared = False,
        linkstatic = False,
        # local_defines = [""],
        # nocopts = "",
        # tags = [""],
        visibility = ["//visibility:public"],
    )