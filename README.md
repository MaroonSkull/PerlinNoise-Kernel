# PerlinNoise
Simple project where I'm trying to create Perlin noise generation algorithm from scratch using nvidia CUDA.  
Also, in this project I'm learning how to work with OpenGL.

List of targets
==

CUDA
--
- [x] 1D Perlin noise
  - [x] Noise generation
  - [x] Putting octaves on noise
    - [x] ~~Naïve implementation~~ (Removed as unnecessary)
    - [x] Shared memory* implementation
      - [x] Naïve implementation
      - [x] Implementation optimized for any noise size

- [ ] 2D Perlin noise
  - [ ] Noise generation
  - [ ] Putting octaves on noise
    - [ ] Naïve implementation
    - [ ] Shared memory implementation

>*shared memory is the GPU equivalent of the L1-cache in the CPU.

OpenGL
--
- [x] displaying 1D Perlin noise...
  - [x] ...as a graph of a function

- [ ] displaying 2D Perlin noise...
  - [ ] ...as a 2d texture
  - [ ] ...as a height map

- [ ] make it interactive

Benchmarking
--
- [ ] Add code to measure the performance of CUDA cores
- [ ] Display the results in the form of a table

>_**planned**_ table view:

| **#** | **kernelName**            | **occupancy[%]** | **perform.[%]** | **time[ms]** | **boost[%]** |
| ----: | :------------------------ | :--------------- | :-------------- | :----------- | :----------- |
| ┌1    | kernel 1                  | 66               | 100             | 0.02199      | 100          |
| └2    | kernel 2 (compare with 1) | 100              | 124.93          | 0.01718      | +17.74       |
| ┌3    | kernel 3                  | 100              | 100             | 0.0228       | 100          |
| ├4    | kernel 4 (compare with 3) | 100              | 106.743         | 0.02136      | 6.32         |
| └5    | kernel 5 (compare with 3) | 100              | 93.097          | 0.0245       | -14.66       |

C++
--
- [ ] Structure for storing benchmarking results
- [ ] Structure for storing the data of the function under test
- [ ] Wrap functions for Perlin noise generation in a class
- [ ] Create an enum with errors provided by the program

Project design
--
- [ ] Add code usage examples
- [ ] Add screenshots with code execution results
- [ ] Add code documentation
- [ ] Translate all comments into English (some of them are written in the native language of the developer - Russian)

Possible plans for project expansion
==
- Add 3D perlin noise
- Add creation of animated 2d noise with a smooth change in the coefficients
- Make it into a library
- Generate 2D noise using CUDA directly into a texture video card, explore the possibility of directly mapping textures using OpenGL
- Add GUI (qt qml?)
- Make the code independent of the used OC (at the moment, the project is oriented towards Windows)
