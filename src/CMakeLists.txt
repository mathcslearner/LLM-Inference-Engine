# Modules, listed in dependency-layer order (see the architecture map in
# CLAUDE.md). Each builds as its own static library target `engine_<module>`
# with alias `engine::<module>`.
add_subdirectory(core)
add_subdirectory(tensor)
add_subdirectory(memory)
add_subdirectory(cuda)
add_subdirectory(kernels)
add_subdirectory(cpu)
add_subdirectory(model)
add_subdirectory(tokenizer)
add_subdirectory(kvcache)
add_subdirectory(sampling)
add_subdirectory(quant)
add_subdirectory(spec)
add_subdirectory(distributed)
add_subdirectory(engine)
add_subdirectory(scheduler)
add_subdirectory(runtime)
add_subdirectory(server)
add_subdirectory(metrics)

# Placeholder entry point; replaced by the real server binary in M9.
add_executable(engine_main main.cpp)
set_target_properties(engine_main PROPERTIES OUTPUT_NAME engine)
target_compile_definitions(engine_main PRIVATE ENGINE_VERSION="${PROJECT_VERSION}")
target_link_libraries(engine_main PRIVATE engine_warnings engine::core)
