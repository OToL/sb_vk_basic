from subprocess import Popen, PIPE
import sys
import os
import os.path

try:  
   vulkan_sdk_dir = os.environ["VULKAN_SDK"]
except KeyError: 
   print("Please set the environment variable FOO")
   sys.exit(1)

curr_dir_path = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(curr_dir_path, "..", "data")
gsllang_validtor = os.path.join(vulkan_sdk_dir, "bin", "glslangValidator")

shader_build_process = Popen([gsllang_validtor, "-V", "default.vert"], stdout=PIPE, stderr=PIPE, cwd=data_dir)
(output, err) = shader_build_process.communicate()
shader_build_process.wait()

shader_build_process = Popen([gsllang_validtor, "-V", "default.frag"], stdout=PIPE, stderr=PIPE, cwd=data_dir)
(output, err) = shader_build_process.communicate()
shader_build_process.wait()
