#include "Perlin.hpp"
#include "cuda/api/launch_configuration.hpp"
#include <iostream>

#ifdef __CUDACC__
// #include <cuda/api/detail/unique_span.hpp>
#include <cuda/api.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/memory.hpp>

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>

namespace kernels {

/**
 * @brief Линейная интерполяция.
 *
 * @details Функция вычисляет значение в точке t на промежутке [0, 1] между
 * двумя прямыми с наклонами k0 и k1 соответственно.
 *
 * @param k0 Значение наклона прямой в точке 0.
 * @param k1 Значение наклона прямой в точке 1.
 * @param t Точка, значение в которой интерполируется.
 *
 * @return Результат интерполяции.
 */
template <typename T>
__device__ __forceinline__ T lerp_kernel(T k0, T k1, T t) {
  // (1-t)*k0 + t*k1 = k0 - t*k0 + t*k1 = t*(k1 - k0) + k0
  return fma(t, k1 - k0, k0);
}

/**
 * @brief Сигмоидальная функция из семейства smoothstep.
 *
 * @details Используется для создания более интенсивного градиента шума.
 * Оригинальный первый полином.
 *
 * @param x Значение градиента (он же t).
 *
 * @return возвращает классический smoothstep(x).
 *
 * @see https://en.wikipedia.org/wiki/Smoothstep#Variations
 */
template <typename T> __device__ __forceinline__ T smoothstep_kernel(T x) {
  // 3 * x^2 - 2 * x^3 = -x * x * (2 * x - 3);
  return fma(static_cast<T>(2), x, static_cast<T>(-3)) * -x * x;
}

/**
 * @brief Сигмоидальная функция из семейства smoothstep.
 *
 * @details Используется для создания ещё более интенсивного градиента шума.
 * Оригинальный второй полином Кена Перлина.
 *
 * @see https://en.wikipedia.org/wiki/Smoothstep#Variations
 *
 * @param x Значение градиента (он же t).
 *
 * @return Возвращает классический smootherstep(x).
 */
template <typename T> __device__ __forceinline__ T smootherstep_kernel(T x) {
  // 6x^5 - 15x^4 + 10x^3 = x^3(6x^2 - 15x + 10)
  return fma(static_cast<T>(6), x * x,
             fma(static_cast<T>(-15), x, static_cast<T>(10))) *
         x * x * x;
}

/**
 * Вычисление одномерного шума Перлина.
 * Вычисляет массив любой длины, допустимой видеокартой
 * (для CC3.0+ это (2^31 − 1)*2^10 ≈ 2.1990233e+12 значений)
 *
 * \par[ret] noise – массив с результатом вычисления шума перлина на оси.
 * \par[ret] octave – массив для хранения первой октавы шума Перлина.
 * \param gradients – массив со значениями наклона уравнений в контрольных
 * узлах.
 * \param axisLength – длина массива noise.
 * \param axisStep – величина шага между точками, в которых вычисляется шум.
 * \param pointsBetweenGradients – количество точек между контрольными узлами.
 */
template <typename T>
__global__ void Perlin1D_kernel(T *noise, const T *gradients,
                                uint32_t axisLength, T axisStep,
                                uint32_t pointsBetweenGradients) {
  // количество threads, выполняющих вычисления
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= axisLength)
    return;

  // 0 0 0 / 1 1 1 / 2 2 2 / .. – какие точки шума к каким контрольным узлам
  // принадлежат
  uint32_t n = id * axisStep;

  // 0 1 2 / 0 1 2 / 0 1 2 / .. – позиция точки между левым и правым контрольным
  // узлом
  uint32_t dotNum = id % (pointsBetweenGradients + 1);

  // 0.0 0.33 0.66 / 0 0.33 0.66 / .. – численное значение точки для
  // интерполяции
  T t = dotNum * axisStep;

  // Применяем сигмоидальную(на промежутке [0, 1]) функцию, реализуя градиент
  t = smoothstep_kernel(t);

  // kx+b (b = 0)
  T y0 = gradients[n] * t;

  // kx+b (b = -k) = k(x-1)
  T y1 = gradients[n + 1] * (t - 1);

  // Интерполяцией находим шум, пишем сразу в выходной массив
  noise[id] = lerp_kernel(y0, y1, t);
}

template <typename T> __device__ T pow_func(T base, int exponent) {
  T result = 1;
  for (int i = 0; i < exponent; ++i) {
    result *= base;
  }
  return result;
}

// Основное ядро для октав
template <typename T>
__global__ void ApplyOctavesKernel(const T *input, T *output, int size,
                                   int numOctaves, T lacunarity,
                                   T persistence) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size)
    return;

  T value = input[id]; // Исходное значение шума

  for (int octave = 0; octave < numOctaves; ++octave) {
    T frequency = pow_func(lacunarity, octave + 1);
    T amplitude = pow_func(persistence, octave + 1);

    int index = static_cast<int>(id * frequency) % size;

    T sample;
    if (index + 1 >= size) {
      sample = input[size - 1];
    } else {
      T fract = id * frequency - floor(id * frequency);
      sample = lerp_kernel<float>(input[index], input[index + 1], fract);
    }

    value += amplitude * sample;
  }

  output[id] = value;
}

/**
 * Накладывает на готовый одномерный шум Перлина указанное количество октав.
 * Данная версия алгоритма предполагает, что в разделяемую память полностью
 * помещается первая октава. Это позволяет вычислять октавы для шума fp64 длиной
 * вплоть до 8192, либо fp32 до 16384 значений.
 *
 * \param noise – массив с результатом наложения октав на шум Перлина на оси.
 * \param octave – массив для хранения первой октавы шума Перлина.
 * \param size – количество изменяемых значений шума, длина массива noise.
 * \param octaveNum – количество октав.
 *
 * \return noise – функция изменяет переданный массив (хранится в памяти GPU).
 */
template <typename T>
__global__ void Perlin1Doctave_shared_kernel(T *noise, const T *octave,
                                             uint32_t size,
                                             uint32_t octaveNum) {
  // выделяем разделяемую память для октав.
  /* используем 32KB памяти, на всех более-менее современных архитектурах
   * (CC 3.7+) именно такое значение позволит запускать минимум 2 блока на одном
   * sm. Это приведёт к потенциальной 100% занятости устройства. Так же это даёт
   * 8192 fp32 значения, или 4096 fp64. */
  constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
  __shared__ T sharedOctave[sharedOctaveLength];

  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  // if(id >= size) return; // здесь не делаем, потому что неиспользуемые сейчас
  // потоки могут пригодиться в обработке других блоков

  if (size > 2 * sharedOctaveLength) { // проверка на неправильный вызов ядра
    if (id == 0)
      printf("size = %d, 2*sharedOctaveLength = %d. exit.\r\n\r\n", size,
             2 * sharedOctaveLength);
    return;
  }

  // Сохраняем в разделяемой памяти блока первую октаву шума.
  /* Нам необходимо, чтобы каждый блок имел локальную копию первой октавы,
   * поэтому каждый блок в цикле копирует в свою разделяемую память октаву
   * из глобальной памяти последовательно. Это наиболее оптимизированный
   * режим чтения данных из глобальной памяти в разделяемую (coalesced).
   * Каждый поток в блоке выполнит операцию копирования вплоть до maxI раз,
   * где maxI = размер октавы / размер блока, округлённое вверх до целого.
   * Заметим, что maxI - это не что иное, как количество блоков в сетке,
   * раздёлённое на 2 с округлением вверх. Деление n на d с округлением
   * быстрее всего реализовать с помощью нехитрого преобразования:
   * (n+d-1)/d. Поскольку мы делим на 2, можно записать: (n+2-1)/2 = (n+1)/2.
   * Деление на 2, как всем известно, можно заменить на битовый сдвиг.
   * Так вычисление maxI можно заменить на (gridDim.x + 1) >> 1 */
  for (uint32_t i = 0; i < (gridDim.x + 1) >> 1; i++) {
    uint32_t sharedId = blockDim.x * i + threadIdx.x;
    if (size > 2 * sharedId) // контроллируем выход за пределы массива
      sharedOctave[sharedId] = octave[sharedId];
  }

  // Синхронизируем выполнение на уровне блока.
  __syncthreads();
  // На этом моменте вся первая октава записана в разделяемую память данного
  // блока

  // Применяем наложение октав, каждый раз основываясь на предыдущей октаве
  for (int j = 1; j <= octaveNum; j++) {
    int octavePov = 1 << j;
#pragma unroll
    for (int i = 0; i < octavePov;
         i++) { // здесь мб будет смысл запихнуть if(выполнился поток) break;,
                // забенчить потом
      if ((id >= i * size / octavePov) && (id < (i + 1) * size / octavePov)) {
        noise[id] +=
            sharedOctave[(id - i * size / octavePov) * (octavePov >> 1)] /
            (octavePov >> 1);
        break;
      }
    }
  }
}

/**
 * Накладывает на готовый одномерный шум Перлина указанное количество октав.
 * Данная версия алгоритма позволяет накладывать на шум октавы произвольной
 * длины.
 *
 * \param noise – массив с результатом наложения октав на шум Перлина на оси.
 * \param octave – массив для хранения первой октавы шума Перлина.
 * \param size – количество изменяемых значений шума, длина массива noise.
 * \param octaveNum – количество октав.
 *
 * \return noise – функция изменяет переданный массив (хранится в памяти GPU).
 */
template <typename T>
__global__ void
Perlin1Doctave_shared_unlimited_kernel(T *noise, const T *octave, uint32_t size,
                                       uint32_t octaveNum) {
  // выделяем разделяемую память для октав.
  /* используем 32KB памяти, на всех более-менее современных архитектурах
   * (CC 3.7+) именно такое значение позволит запускать минимум 2 блока на одном
   * sm. Это приведёт к потенциальной 100% занятости устройства. Так же это даёт
   * 8192 fp32 значения, или 4096 fp64. */
  constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
  __shared__ T sharedOctave[sharedOctaveLength];

  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size)
    return;

  /* Повторяем вычисления, каждый раз обрабатывая ту часть октавы, которая
   * помещается в разделяемую память Повторить вычисления придётся
   * numOfOctaveCalc раз, где numOfOctaveCalc = ceil(размер октавы/размер
   * разделяемой памяти) = ceil(ceil(size/2) / sharedOctaveLength) = [ceil(a/b)
   * = floor((a+b-1)/b)] = ceil(floor((size+1) / 2) / sharedOctaveLength) =
   * floor((floor((size+1) / 2) + sharedOctaveLength - 1) / sharedOctaveLength)
   * = = (((size+1) >> 1) + sharedOctaveLength - 1) / sharedOctaveLength;*/
  uint32_t numOfOctaveCalc =
      (((size + 1) >> 1) + sharedOctaveLength - 1) / sharedOctaveLength;
  for (uint32_t i = 0; i < numOfOctaveCalc; i++) {
    // Сохраняем в разделяемой памяти часть первой октавы шума.

    // Защита (для каждого цикла после первого) - ждём, пока все операции с
    // разделяемой памятью закончатся, перед тем, как её менять.
    __syncthreads();

    // Нам необходимо, чтобы каждый блок имел локальную копию части первой
    // октавы
    // uint32_t control = sharedOctaveLength < (size + 1) >> 1 ?
    // sharedOctaveLength : (size + 1) >> 1; // учитываем оба случая, когда мало
    // разделяемой или когда мало шума
    uint32_t maxJ = (sharedOctaveLength + blockDim.x - 1) / blockDim.x;
    for (uint32_t j = 0; j < maxJ; j++) {
      uint32_t globalId = sharedOctaveLength * i + blockDim.x * j +
                          threadIdx.x; // тут min(blockDim, realDim)
      uint32_t sharedId = blockDim.x * j + threadIdx.x;
      if (sharedId <
          sharedOctaveLength) // контроллируем выход за пределы массива
        sharedOctave[sharedId] = octave[globalId];
    }

    // Синхронизируем выполнение на уровне блока.
    __syncthreads();
    // На этом моменте вся часть первой октавы, которая помещается в разделяемую
    // память, записана в неё

    // Применяем наложение октав, каждый раз основываясь на предыдущей октаве
    for (int j = 1; j <= octaveNum; j++) {
      int octavePov = 1 << j;
#pragma unroll
      for (int k = 0; k < octavePov; k++) {
        uint32_t globalMin = k * size / octavePov;
        uint32_t globalMax = (k + 1) * size / octavePov;
        uint32_t sharedMin =
            sharedOctaveLength / (octavePov / 2) * i + globalMin;
        uint32_t sharedMax =
            sharedOctaveLength / (octavePov / 2) * (i + 1) + globalMin;
        int32_t sharedId = id - sharedMin;
        if ((id >= globalMin) && (id < globalMax) && (id >= sharedMin) &&
            (id < sharedMax)) {
          if (sharedId * (octavePov >> 1) >= sharedOctaveLength)
            printf("outOfRangeAdress! k = %d, id = %d, sharedId = %d, "
                   "octavePov>>1 = %d\r\n",
                   k, id, sharedId, octavePov >> 1);
          else {
            noise[id] +=
                sharedOctave[sharedId * (octavePov >> 1)] / (octavePov >> 1);
            break;
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void Perlin1Dvertices_kernel(T *vertices, const T *noise,
                                        uint32_t size) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id >= size)
    return;

  vertices[2 * id] = 2.0 / size * static_cast<float>(id) - 1.0;
  vertices[2 * id + 1] = noise[id];
  if (id == size - 1) {
    vertices[2 * id + 2] = 1.;
    vertices[2 * id + 3] = 0.;
  }
}

} // namespace kernels

/**
 * @brief Реализация шума Перлина с использованием NVidia CUDA
 */
class PerlinImpl : public IPerlin {

  using i = IPerlin;

  /// Линеаризованный массив градиентов
  std::vector<float> hv_gradients_;
  thrust::host_vector<float> h_gradients_;
  thrust::device_vector<float> d_gradients_;

  /// Значения шума во всех точках пространства
  std::vector<float> hv_noise_;
  thrust::host_vector<float> h_noise_;
  thrust::device_vector<float> d_noise_;
  thrust::device_vector<float> d_tempNoise_;

  void calculateNoise() {
    h_noise_.clear();
    hv_noise_.clear();

    // Количество точек пространства - это длина одной оси в степени измерений.
    int numberOfPoints = std::pow(i::getAxisLenght(), i::numberOfDimensions_);
    int sizeOfPoint = i::numberOfDimensions_;

    d_noise_.resize(sizeOfPoint * numberOfPoints);

    if (i::numberOfDimensions_ == 1) {
      // Получаем raw-указатели на device данные
      float *d_noise = thrust::raw_pointer_cast(d_noise_.data());
      const float *d_gradients = thrust::raw_pointer_cast(d_gradients_.data());

      auto launch_config =
          cuda::launch_config_builder()
              .block_size(numberOfPoints > 256 ? 256 : numberOfPoints)
              .grid_size(numberOfPoints > 256 ? (numberOfPoints + 255) / 256
                                              : 1)
              .build();

      // std::cout << "CUDA kernel launch with " <<
      // launch_config.dimensions.grid.x
      //           << " blocks of " << launch_config.dimensions.block.x
      //           << " threads each\n";

      // Launch a kernel on the GPU with one thread for each element.
      cuda::launch(kernels::Perlin1D_kernel<float>, launch_config, d_noise,
                   d_gradients, numberOfPoints,
                   i::getDistanceBetweenTwoPoints(),
                   i::pointsBetweenGradients_);

      if (i::numberOfOctaves_)
        applyOctaves(launch_config);

      if (cuda::outstanding_error::get() != cuda::status_t::CUDA_SUCCESS) {
        std::cerr << "CUDA error occurred during kernel execution\n";
        return;
      }

      // Проверка ошибок ядра
      cudaError_t kernelError = cudaGetLastError();
      if (kernelError != cudaSuccess) {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(kernelError)
                  << std::endl;
      }

      // Синхронизация для корректного завершения
      cudaDeviceSynchronize();
    }

    h_noise_ = d_noise_;
    hv_noise_.resize(h_noise_.size());
  }

  void applyOctaves(cuda::launch_configuration_t &kernels_config) {
    d_tempNoise_ = d_noise_;

    // Получаем raw-указатели на device данные
    float *d_noise = thrust::raw_pointer_cast(d_noise_.data());
    float *d_tempNoise = thrust::raw_pointer_cast(d_tempNoise_.data());

    // Количество точек пространства - это длина одной оси в степени измерений.
    int numberOfPoints = std::pow(i::getAxisLenght(), i::numberOfDimensions_);

    // Launch a kernel on the GPU with one thread for each element.
    cuda::launch(kernels::ApplyOctavesKernel<float>, kernels_config, d_noise,
                 d_tempNoise, numberOfPoints, i::numberOfOctaves_,
                 i::lacunarity_, i::persistence_);

    if (cuda::outstanding_error::get() != cuda::status_t::CUDA_SUCCESS) {
      std::cerr << "CUDA error occurred during octaves kernel execution\n";
      return;
    }

    d_noise_ = d_tempNoise_;
  }

  void addGradients() {
    static thrust::default_random_engine rng(1337);
    static thrust::uniform_real_distribution<float> dist(-1.0, 1.0);

    if (!hv_gradients_.empty()) {
      hv_gradients_.insert(hv_gradients_.end() - 1, dist(rng));
    } else {
      // Если массив пуст, добавляем 0, чтобы соблюсти условие
      auto gradient = dist(rng);
      hv_gradients_.push_back(gradient);
      hv_gradients_.push_back(gradient);
    }
  }

  void removeGradient() {
    if (!hv_gradients_.empty()) {
      hv_gradients_.erase(hv_gradients_.end() - 2);
    }
  }

public:
  static const bool isCUDAAvailable{true};

  /**
   * @brief Конструктор класса Perlin.
   */
  PerlinImpl() : IPerlin() { std::cout << "Use CUDA" << std::endl; }

  // Инициализация градиентов случайными значениями
  std::vector<float> &initializeGradients() {
    for (auto i = hv_gradients_.size(); i <= i::numberOfGradients_;
         i = hv_gradients_.size()) {
      addGradients();
    }
    for (auto i = hv_gradients_.size() - 1; i > i::numberOfGradients_;
         i = hv_gradients_.size() - 1) {
      removeGradient();
    }

    hv_gradients_.back() = hv_gradients_.front(); // Замыкаем циклический массив

    h_gradients_ = hv_gradients_;
    d_gradients_ = h_gradients_;

    hv_noise_.resize(i::getAxisLenght());
    h_noise_.resize(i::getAxisLenght());
    d_noise_.resize(i::getAxisLenght());

    return hv_gradients_;
  }

  std::vector<float> &getNoise() {
    initializeGradients();
    calculateNoise();
    thrust::copy(h_noise_.begin(), h_noise_.end(), hv_noise_.begin());
    return hv_noise_;
  }
};

#else

#include <cmath>
#include <random>

class PerlinImpl : public IPerlin {

  using i = IPerlin;

  /// Линеаризованный массив градиентов
  std::vector<float> gradients_;

  /// Значения шума во всех точках пространства
  std::vector<float> noise_;

  /**
   * @brief Линейная интерполяция.
   *
   * @details Функция вычисляет значение в точке t на промежутке [0, 1] между
   * двумя прямыми с наклонами k0 и k1 соответственно.
   *
   * @param k0 Значение наклона прямой в точке 0.
   * @param k1 Значение наклона прямой в точке 1.
   * @param t Точка, значение в которой интерполируется.
   *
   * @return Результат интерполяции.
   */
  float lerp(float k0, float k1, float t) {
    // (1-t)*k0 + t*k1 = k0 - t*k0 + t*k1 = t*(k1 - k0) + k0
    return fma(t, k1 - k0, k0);
  }

  /**
   * @brief Сигмоидальная функция из семейства smoothstep.
   *
   * @details Используется для создания более интенсивного градиента шума.
   * Оригинальный первый полином.
   *
   * @param x Значение градиента (он же t).
   *
   * @return возвращает классический smoothstep(x).
   *
   * @see https://en.wikipedia.org/wiki/Smoothstep#Variations
   */
  float smoothstep(float x) {
    // 3 * x^2 - 2 * x^3 = -x * x * (2 * x - 3);
    return fma(static_cast<float>(2), x, static_cast<float>(-3)) * -x * x;
  }

  /**
   * @brief Сигмоидальная функция из семейства smoothstep.
   *
   * @details Используется для создания ещё более интенсивного градиента шума.
   * Оригинальный второй полином Кена Перлина.
   *
   * @see https://en.wikipedia.org/wiki/Smoothstep#Variations
   *
   * @param x Значение градиента (он же t).
   *
   * @return Возвращает классический smootherstep(x).
   */
  float smootherstep(float x) {
    // 6x^5 - 15x^4 + 10x^3 = x^3(6x^2 - 15x + 10)
    return fma(static_cast<float>(6), x * x,
               fma(static_cast<float>(-15), x, static_cast<float>(10))) *
           x * x * x;
  }

  /**
   * @brief Вычисление одномерного шума Перлина.
   */
  void calculateBaseNoise() {
    const auto noiseSize = i::getAxisLenght();
    const auto axisStep = i::getDistanceBetweenTwoPoints();
    for (int id = 0; id < noiseSize; id++) {
      // 0 0 0 / 1 1 1 / 2 2 2 / .. – какие точки шума к каким контрольным узлам
      // принадлежат
      int n = id * axisStep;

      // 0 1 2 / 0 1 2 / 0 1 2 / .. – позиция точки между левым и правым
      // контрольным узлом
      uint32_t dotNum = id % (i::pointsBetweenGradients_ + 1);

      // 0.0 0.33 0.66 / 0 0.33 0.66 / .. – численное значение точки для
      // интерполяции
      float t = dotNum * axisStep;

      // Применяем сигмоидальную(на промежутке [0, 1]) функцию, реализуя
      // градиент
      t = smoothstep(t);

      // kx+b (b = 0)
      float y0 = gradients_[n] * t;

      // kx+b (b = -k) = k(x-1)
      float y1 = gradients_[n + 1] * (t - 1);

      // Интерполяцией находим шум, пишем сразу в выходной массив
      noise_[id] = lerp(y0, y1, t);
    }
  }

  void applyOctaves() {
    std::vector<float> tempNoise = noise_;
    for (int i = 1; i <= i::numberOfOctaves_; ++i) {
      float frequency = powf(i::lacunarity_, i);
      float amplitude = powf(i::persistence_, i);
      for (int j = 0; j < noise_.size(); ++j) {
        int index = static_cast<int>(j * frequency) % noise_.size();
        float sample;
        if (index + 1 >= noise_.size()) {
          sample = noise_[noise_.size() - 1];
        } else {
          float fract = j * frequency - floorf(j * frequency);
          sample = lerp(noise_[index], noise_[index + 1], fract);
        }
        tempNoise[j] += amplitude * sample;
      }
    }
    noise_ = tempNoise;
  }

  void addGradients() {
    static std::default_random_engine rng(1337);
    static std::uniform_real_distribution dist(-1.0, 1.0);

    if (!gradients_.empty()) {
      gradients_.insert(gradients_.end() - 1, dist(rng));
    } else {
      // Если массив пуст, добавляем 0, чтобы соблюсти условие
      auto gradient = dist(rng);
      gradients_.push_back(gradient);
      gradients_.push_back(gradient);
    }
  }

  void removeGradient() {
    if (!gradients_.empty()) {
      gradients_.erase(gradients_.end() - 2);
    }
  }

public:
  PerlinImpl() : IPerlin() { std::cout << "Use CPU" << std::endl; }

  // Инициализация градиентов случайными значениями
  std::vector<float> &initializeGradients() {
    for (auto i = gradients_.size(); i <= i::numberOfGradients_;
         i = gradients_.size()) {
      addGradients();
    }
    for (auto i = gradients_.size() - 1; i > i::numberOfGradients_;
         i = gradients_.size() - 1) {
      removeGradient();
    }

    gradients_.back() = gradients_.front(); // Замыкаем циклический массив

    noise_.resize(i::getAxisLenght());

    return gradients_;
  }

  std::vector<float> &getNoise() {
    initializeGradients();
    calculateBaseNoise();
    if (i::numberOfOctaves_)
      applyOctaves();
    return noise_;
  }
};

#endif // __CUDACC__

std::unique_ptr<IPerlin> PimplPerlinFactory::createPerlin() {
  return std::make_unique<PerlinImpl>();
}
