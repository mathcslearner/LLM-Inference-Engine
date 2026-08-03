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
# Emitted into <build>/bin: the default (<build>/src/) would collide with the
# <build>/src/engine/ directory that the engine module's subdirectory creates.
set_target_properties(engine_main PROPERTIES
  OUTPUT_NAME engine
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
target_compile_definitions(engine_main PRIVATE ENGINE_VERSION="${PROJECT_VERSION}")
# fmt linked here as a smoke test that fetched dependencies compile and link
# under the project toolchain; broader use starts with logging in M0-T06.
target_link_libraries(engine_main PRIVATE engine_warnings engine::core fmt::fmt)
