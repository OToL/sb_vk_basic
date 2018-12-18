#!/bin/bash
ini_working_dir=$PWD
working_dir=$ini_working_dir/../data
cd $working_dir

${VULKAN_SDK}/bin/glslangValidator -V default.vert
${VULKAN_SDK}/bin/glslangValidator -V default.frag

cd $ini_working_dir