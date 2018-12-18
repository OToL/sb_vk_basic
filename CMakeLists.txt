project(sb_vk_triangle LANGUAGES CXX)

find_package(vulkan REQUIRED)
find_library(carbon Carbon)
find_library(cocoa Cocoa)
find_library(iokit IOKit)
find_library(quartzCore QuartzCore)

sbAddExecutable(sb_vk_triangle
    src/main.cpp)

target_include_directories(sb_vk_triangle
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/thirdparty/inc)

target_compile_options(sb_vk_triangle
    PRIVATE 
        -Wno-zero-as-null-pointer-constant
        -Wno-unused-parameter
        -Wno-unused-variable)

target_link_libraries(sb_vk_triangle 
    PRIVATE
        sb::core
        Vulkan::Vulkan
        ${CMAKE_CURRENT_LIST_DIR}/thirdparty/lib/libglfw3.a
        ${carbon}
        ${cocoa}
        ${iokit}
        ${quartzCore})
