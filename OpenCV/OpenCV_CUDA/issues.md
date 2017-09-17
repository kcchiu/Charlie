# Issues encountered using OpenCv with CUDA
Environment:  
OS: Windows 10 Home 64 bit  
CUDA: 8.0  
OpenCV:  3.3.0 vc-14  
Visual Studio:  14 2015 Enterprise  
CMake: 3.9.2  
GPU: Nvidia Quadro M1200  

For the official guide to build OpenCV with CUDA, please follow [here](http://docs.opencv.org/master/d3/d52/tutorial_windows_install.html).  

## 1. Building with Visual Studio 2015
### Issue
Failing to build OpenCV with CUDA  
### Solution
1. In CMake, enable `WITH_CUBLAS` aswell as `WITH_CUDA`  
2. Turn off `BUILD_PERF_TESTS` and `BUILD_TESTS` 
3. `Configure` and `Generate`.  
[Reference](https://github.com/opencv/opencv/issues/7992)  
### Issue
Unable to find `python36_d.lib`  
### Solution
__Option 1:__ Build the library yourself  
1. Download the source tarball from <http://www.python.org/download>  
2. Extract the tarball (7zip will do the trick) and go into the resulting directory (should be something like Python-3.3.2).  
3. From the Python directory, go to the PCBuild folder. There are two important files here: readme.txt, which contains the instructions for building Python in Windows (even if it uses the UNIX line feed style...), and pcbuild.sln, which is the Visual Studio solution that builds Python.  
4. Open pcbuild.sln in Visual Studio. (I am assuming you are using Visual Studio 10; readme.txt contains specific instructions for older versions of Visual Studio.)  
5. Make sure Visual Studio is set to the "debug" configuration, and then build the solution for your appropriate architecture (x64 or Win32). You may get a few failed subprojects, but not all of them are necessary to build python33_d. By my count, 8 builds failed and I got a working .lib file anyway.   
6. You will find python33_d.lib and python33_d.dll in either the PCBuild folder (if building Win32) or the amd64 subfolder (if building x64).   

__Option 2:__ Use customized installation  
If you install python via the installers on python.org, you can tell the installer to include the debugging symbols and binaries such as the pythonXX_d.dll file by selecting `Customize Installation` while installing. This may be the easiest solution if you're not very savvy at building the project yourself (like me). Too bad I don't see any way to do this with the anaconda distribution.  
