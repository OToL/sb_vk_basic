from subprocess import Popen, PIPE
import sys
import os
import os.path
import shutil

def buildShader(glslc, input, output):
   print("Buildling shader {0} ...".format(input))
   shader_build_process = Popen([glslc, input, "-o", output], stdout=PIPE, stderr=PIPE, cwd=data_dir)
   (output, err) = shader_build_process.communicate()
   shader_build_process.wait()

   if len(output) != 0:
      print(output)

   if len(err) != 0:
      print(err)

if __name__ == "__main__":
   try:  
      vulkan_sdk_dir = os.environ["VULKAN_SDK"]
   except KeyError: 
      print("Please set the environment variable FOO")
      sys.exit(1)

   curr_dir_path = os.path.dirname(os.path.abspath(__file__))
   data_dir = os.path.join(curr_dir_path, "..", "data")
   build_dir = os.path.join(curr_dir_path, "..", "build")
   glslc = os.path.join(vulkan_sdk_dir, "bin", "glslc")

   if not os.path.exists(build_dir):
      os.mkdir(build_dir)

   buildShader(glslc, "basic.vert", os.path.join(build_dir, "basic.vert"))
   buildShader(glslc, "basic.frag", os.path.join(build_dir, "basic.frag"))

   shutil.copyfile(os.path.join(data_dir, "texture.jpg"), os.path.join(build_dir, "texture.jpg"))
   shutil.copyfile(os.path.join(data_dir, "viking_room.png"), os.path.join(build_dir, "viking_room.png"))
   shutil.copyfile(os.path.join(data_dir, "viking_room.obj"), os.path.join(build_dir, "viking_room.obj"))
