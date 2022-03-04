#include "Definitions.h"

/**
* линейная интерполяция точки t на промежутке [0, 1] между двумя прямыми с наклонами k0 и k1 соответственно.
* 
* \param k0 – значение наклона прямой в точке 0.
* \param k1 – значение наклона прямой в точке 1.
* \param t – точка, значение в которой интерполируется.
* 
* \return Результат интерполяции.
*/
template <typename T>
__device__ inline
T lerp_kernel(T k0, T k1, T t) {
	return fma(t, k1 - k0, k0); // (1-t)*k0 + t*k1 = k0 - t*k0 + t*k1 = t*(k1 - k0) + k0
}

/**
* Сигмоидальная функция из семейства smoothstep, используется для создания более интенсивного градиента шума. 
* Подробнее см. https://en.wikipedia.org/wiki/Smoothstep#Variations
* 
* \param x – значение градиента (он же t)
* 
* \return возвращает классический smootherstep(x). Используется оригинальный второй полином Кена Перлина.
*/
template <typename T>
__device__ inline
T smootherstep_kernel(T x) {
	return fma(static_cast<T>(6), x * x, fma(static_cast<T>(-15), x, static_cast<T>(10))) * x * x * x; // 6x^5 - 15x^4 + 10x^3 = x^3(6x^2 - 15x + 10)
}

/**
* Вычисление одномерного шума Перлина.
* Вычисляет массив любой длины, допустимой видеокартой (для CC3.0+ это (2^31 − 1)*2^10 ≈ 2.1990233e+12 значений)
* 
* \param noise – массив с результатом вычисления шума перлина на оси.
* \param octave – массив для хранения первой октавы шума Перлина.
* \param k – массив со значениями наклона уравнений в контрольных узлах.
* \param size – длина массива noise.
* \param step – величина шага между точками, в которых вычисляется шум.
* \param numSteps – количество точек между контрольными узлами.
* \param isOctaveCalkNeed – будут ли в дальнейшем вычисляться октавы.
* 
* \return noise – (см. описание параметра) функция изменяет переданный массив (хранится в памяти GPU).
* \return octave – (см. описание параметра) функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1D_kernel(T *noise, T *octave, const T *k, uint32_t size, T step, uint32_t numSteps, bool isOctaveCalkNeed) {
	uint32_t id = blockIdx.x*blockDim.x+threadIdx.x;// [0..] – всего точек для просчёта
	if(id >= size) return;
	uint32_t n = static_cast<T>(id) * step;			// 0 0 / 1 1 / 2 2 / .. – какие точки к каким контрольным точкам принадлежат
	uint32_t dotNum = id % numSteps;				// 0 1 / 0 1 / 0 1 / .. – какую позицию занимает точка между левой и правой функцией
	T t = dotNum * step;							// 0.33 0.66 / 0.33 0.66 / .. – численное значение точки для интерполяции
	t = smootherstep_kernel<T>(t);					// Применяем сигмоидальную(на промежутке [0, 1]) функцию, реализуя градиент
	T y0 = k[n] * t;								// kx+b (b = 0)
	T y1 = k[n+1] * (t - 1);						// kx+b (b = -k) = k(x-1)
	noise[id] = lerp_kernel<T>(y0, y1, t);			// Интерполяцией находим шум, пишем сразу в выходной массив

	// Если пользователю нужно вычислять октавы, сохраняем в памяти первую окатву шума
	if(isOctaveCalkNeed)
		// Первая октава занимает в два раза меньше памяти, чем исходный шум
		if(id % 2 == 0)
			octave[id >> 1] = noise[id] * 0.5;
}

/**
* Накладывает на готовый одномерный шум Перлина указанное количество октав.
* Данная версия алгоритма предполагает, что в разделяемую память полностью помещается первая октава.
* Это позволяет вычислять октавы для шума fp64 длиной вплоть до 8192, либо fp32 до 16384 значений.
*
* \param noise – массив с результатом наложения октав на шум Перлина на оси.
* \param octave – массив для хранения первой октавы шума Перлина.
* \param size – количество изменяемых значений шума, длина массива noise.
* \param octaveNum – количество октав.
*
* \return noise – функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1Doctave_shared_kernel(T *noise, const T *octave, uint32_t size, uint32_t octaveNum) {
	// выделяем разделяемую память для октав.
	/* используем 32KB памяти, на всех более-менее современных архитектурах (CC 3.7+)
	* именно такое значение позволит запускать минимум 2 блока на одном sm.
	* Это приведёт к потенциальной 100% занятости устройства.
	* Так же это даёт 8192 fp32 значения, или 4096 fp64. */
	constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
	__shared__ T sharedOctave[sharedOctaveLength];

	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if(size > 2*sharedOctaveLength) {
		if (id == 0) printf("size = %d, 2*sharedOctaveLength = %d. exit.\r\n\r\n", size, 2*sharedOctaveLength);
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
	for(uint32_t i = 0; i < (gridDim.x+1) >> 1; i++) {
		uint32_t sharedId = blockDim.x * i + threadIdx.x;
		if(size > 2 * sharedId) // контроллируем выход за пределы массива
			sharedOctave[sharedId] = octave[sharedId];
	}

	// Синхронизируем выполнение на уровне блока.
	__syncthreads();
	// На этом моменте вся первая октава записана в разделяемую память данного блока
	if(id >= size) return;

	// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
	for(int j = 1; j <= octaveNum; j++) {
		int octavePov = 1 << j;
		for(int i = 0; i < octavePov; i++) { // здесь мб будет смысл запихнуть if(выполнился поток) break;, забенчить потом
			if((id >= i * size / octavePov) && (id < (i + 1) * size / octavePov)) {
				noise[id] += sharedOctave[(id - i * size / octavePov) * (octavePov >> 1)] / (octavePov >> 1);
				break;
			}
		}
	}
}

/**
* Накладывает на готовый одномерный шум Перлина указанное количество октав.
* Данная версия алгоритма позволяет накладывать на шум октавы произвольной длины.
*
* \param noise – массив с результатом наложения октав на шум Перлина на оси.
* \param octave – массив для хранения первой октавы шума Перлина.
* \param size – количество изменяемых значений шума, длина массива noise.
* \param octaveNum – количество октав.
*
* \return noise – функция изменяет переданный массив (хранится в памяти GPU).
*/
template <typename T>
__global__
void Perlin1Doctave_shared_unlimited_kernel(T *noise, const T *octave, uint32_t size, uint32_t octaveNum) {
	// выделяем разделяемую память для октав.
	/* используем 32KB памяти, на всех более-менее современных архитектурах (CC 3.7+)
	* именно такое значение позволит запускать минимум 2 блока на одном sm.
	* Это приведёт к потенциальной 100% занятости устройства.
	* Так же это даёт 8192 fp32 значения, или 4096 fp64. */
	constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
	__shared__ T sharedOctave[sharedOctaveLength];

	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	/* Повторяем вычисления, каждый раз обрабатывая ту часть октавы, которая помещается в разделяемую память
	* Повторить вычисления придётся numOfOctaveCalc раз, где numOfOctaveCalc = ceil(размер октавы/размер разделяемой памяти)
	* = ceil(ceil(size/2) / sharedOctaveLength) = [ceil(a/b) = floor((a+b-1)/b)] = ceil(floor((size+1) / 2) / sharedOctaveLength) 
	* = floor((floor((size+1) / 2) + sharedOctaveLength - 1) / sharedOctaveLength) =
	* = (((size+1) >> 1) + sharedOctaveLength - 1) / sharedOctaveLength;*/
	uint32_t numOfOctaveCalc = (((size + 1) >> 1) + sharedOctaveLength - 1) / sharedOctaveLength;
	for(uint32_t i = 0; i < numOfOctaveCalc; i++) {
		// Сохраняем в разделяемой памяти часть первой октавы шума.
		
		// Защита (для каждого цикла после первого) - ждём, пока все операции с разделяемой памятью закончатся, перед тем, как её менять.
		__syncthreads();

		// Нам необходимо, чтобы каждый блок имел локальную копию части первой октавы
		//uint32_t control = sharedOctaveLength < (size + 1) >> 1 ? sharedOctaveLength : (size + 1) >> 1; // учитываем оба случая, когда мало разделяемой или когда мало шума
		uint32_t maxJ = (sharedOctaveLength + blockDim.x - 1) / blockDim.x;
		for(uint32_t j = 0; j < maxJ; j++) {
			uint32_t globalId = sharedOctaveLength * i + blockDim.x * j + threadIdx.x; // тут min(blockDim, realDim)
			uint32_t sharedId = blockDim.x * j + threadIdx.x;
			if(sharedId < sharedOctaveLength) // контроллируем выход за пределы массива
				sharedOctave[sharedId] = octave[globalId];
		}

		// Синхронизируем выполнение на уровне блока.
		__syncthreads();
		// На этом моменте вся часть первой октавы, которая помещается в разделяемую память, записана в неё
		if(id >= size) continue;

		// применяем наложение первой октавы
		/*uint32_t globalMin = 0;
		uint32_t globalMax = size / 2;
		uint32_t sharedMin = sharedOctaveLength * i + globalMin;
		uint32_t sharedMax = sharedOctaveLength * (i + 1) + globalMin;
		uint32_t sharedId = id - sharedMin;

		uint32_t globalMin2 = size / 2;										// 40962
		uint32_t globalMax2 = size;											// 81925
		uint32_t sharedMin2 = sharedOctaveLength * i + globalMin2;			// 40962 - 49154 - 57346 - 65538 - 73730 - 81922
		uint32_t sharedMax2 = sharedOctaveLength * (i + 1) + globalMin2;	// 49154 - 57346 - 65538 - 73730 - 81922 - 90114
		uint32_t sharedId2 = id - sharedMin2;								// 49153-40962=8191
		
		if((id >= globalMin) && (id < globalMax) && (id >= sharedMin) && (id < sharedMax)) {
			if(sharedId >= sharedOctaveLength || sharedId < 0) printf("1, outOfRangeAdress, id = %d, sharedId = %d\r\n", id, sharedId);
			else noise[id] = sharedOctave[sharedId];
		}
		else if((id >= globalMin2) && (id < globalMax2) && (id >= sharedMin2) && (id < sharedMax2)) {
			if(sharedId2 >= sharedOctaveLength || sharedId < 0) printf("2, outOfRangeAdress, id = %d, sharedId = %d\r\n", id, sharedId2);
			else noise[id] = sharedOctave[sharedId2];
		}/**/

		// применяем наложение второй октавы
		/*uint32_t globalMin = 0 * size / 4;
		uint32_t globalMax = 1 * size / 4;
		uint32_t sharedMin = sharedOctaveLength / 2 * i + globalMin;
		uint32_t sharedMax = sharedOctaveLength / 2 * (i + 1) + globalMin;
		uint32_t sharedId = id - sharedMin;

		uint32_t globalMin2 = 1 * size / 4;
		uint32_t globalMax2 = 2 * size / 4;
		uint32_t sharedMin2 = sharedOctaveLength / 2 * i + globalMin2;
		uint32_t sharedMax2 = sharedOctaveLength / 2 * (i + 1) + globalMin2;
		uint32_t sharedId2 = id - sharedMin2;

		uint32_t globalMin3 = 2 * size / 4;
		uint32_t globalMax3 = 3 * size / 4;
		uint32_t sharedMin3 = sharedOctaveLength / 2 * i + globalMin3;
		uint32_t sharedMax3 = sharedOctaveLength / 2 * (i + 1) + globalMin3;
		uint32_t sharedId3 = id - sharedMin3;

		uint32_t globalMin4 = 3 * size / 4;
		uint32_t globalMax4 = 4 * size / 4;
		uint32_t sharedMin4 = sharedOctaveLength / 2 * i + globalMin4;
		uint32_t sharedMax4 = sharedOctaveLength / 2 * (i + 1) + globalMin4;
		uint32_t sharedId4 = id - sharedMin4;

		if((id >= globalMin) && (id < globalMax) && (id >= sharedMin) && (id < sharedMax)) {
			if(sharedId*2 >= sharedOctaveLength) printf("1, outOfRangeAdress, id = %d, sharedId = %d\r\n", id, sharedId * 2);
			else noise[id] = sharedOctave[sharedId*2] / 2;
		}
		else if((id >= globalMin2) && (id < globalMax2) && (id >= sharedMin2) && (id < sharedMax2)) {
			if(sharedId2 * 2 >= sharedOctaveLength) printf("2, outOfRangeAdress, id = %d, sharedId = %d\r\n", id, sharedId2 * 2);
			else noise[id] = sharedOctave[sharedId2 * 2] / 2;
		}
		else if((id >= globalMin3) && (id < globalMax3) && (id >= sharedMin3) && (id < sharedMax3)) {
			if(sharedId3 * 2 >= sharedOctaveLength) printf("3, outOfRangeAdress, id = %d, sharedId = %d\r\n", id, sharedId3 * 2);
			else noise[id] = sharedOctave[sharedId3 * 2] / 2;
		}
		else if((id >= globalMin4) && (id < globalMax4) && (id >= sharedMin4) && (id < sharedMax4)) {
			if(sharedId4 * 2 >= sharedOctaveLength) printf("4, outOfRangeAdress, id = %d, sharedId = %d\r\n", id, sharedId4 * 2);
			else noise[id] = sharedOctave[sharedId4 * 2] / 2;
		}/**/

		// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
		for(int j = 1; j <= octaveNum; j++) {
			int octavePov = 1 << j;
			for(int k = 0; k < octavePov; k++) {
				uint32_t globalMin = k * size / octavePov;
				uint32_t globalMax = (k + 1) * size / octavePov;
				uint32_t sharedMin = sharedOctaveLength / (octavePov/2) * i + globalMin;
				uint32_t sharedMax = sharedOctaveLength / (octavePov/2) * (i + 1) + globalMin;
				int32_t sharedId = id - sharedMin;
				if((id >= globalMin) && (id < globalMax) && (id >= sharedMin) && (id < sharedMax)) {
					if(sharedId * (octavePov >> 1) >= sharedOctaveLength) printf("outOfRangeAdress! k = %d, id = %d, sharedId = %d, octavePov>>1 = %d\r\n", k, id, sharedId, octavePov >> 1);
					else {
						noise[id] += sharedOctave[sharedId * (octavePov >> 1)] / (octavePov >> 1);
						break;
					}
				}
			}
		}/**/
	}
}

/**
* Вспомогательная функция для вычисления шума Перлина на оси с использованием GPU.
*
* \param noise – массив с результатом вычисления шума перлина на оси.
* \param k – массив со значениями наклона уравнений в контрольных узлах.
* \param step – величина шага между точками, в которых вычисляется шум.
* \param numSteps – количество точек между контрольными узлами.
* \param controlPoints – количество узлов.
* \param resultDotsCols - количество точек для просчёта.
* \param octaveNum - количество накладывающихся октав на шум.
*
* \return noise – функция изменяет переданный массив.
* \return cudaError_t
*/
template<typename T>
cudaError_t Perlin1DWithCuda(T *noise, const T *k, T step, uint32_t numSteps, uint32_t controlPoints, uint32_t resultDotsCols, uint32_t octaveNum) {
	T *dev_noise = nullptr; // pointer to noiseult array in VRAM
	T *dev_octave = nullptr; // pointer to temp array in VRAM
	T *dev_k = nullptr; // pointer to array with tilt angle (tg slope angle) in VRAM
	cudaError_t cudaStatus;

	// Choose which GPU to run on.
	cudaStatus = cudaSetDevice(0);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\r\n";
		goto Error;
	}

	// Allocate GPU buffers for arrays.
	cudaStatus = cudaMalloc((void **)&dev_noise, resultDotsCols * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMalloc failed!\r\n";
		goto Error;
	}

	// Массив для октав займёт в 2 раза меньше памяти.
	if(octaveNum > 0) {
		cudaStatus = cudaMalloc((void **)&dev_octave, resultDotsCols * sizeof(T) / 2);
		if(cudaStatus != cudaSuccess) {
			std::cout << stderr << ": cudaMalloc failed!\r\n";
			goto Error;
		}
	}

	cudaStatus = cudaMalloc((void **)&dev_k, controlPoints * sizeof(T));
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMalloc failed!\r\n";
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_k, k, controlPoints * sizeof(T), cudaMemcpyHostToDevice);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMemcpy failed!\r\n";
		goto Error;
	}

	dim3 threadsPerBlock(resultDotsCols, 1, 1);
	dim3 blocksPerGrid(1, 1, 1);

	// 256 взято с потолка из каких-то общих соображений, забейте.
	if(resultDotsCols > 256) {
		threadsPerBlock.x = 256;
		blocksPerGrid.x = (resultDotsCols % 256 == 0) ? resultDotsCols / 256 : resultDotsCols / 256 + 1;
	}

	// Launch a kernel on the GPU with one thread for each element.
	Perlin1D_kernel<T> <<<blocksPerGrid, threadsPerBlock>>>
		(dev_noise, dev_octave, dev_k, resultDotsCols, step, numSteps, static_cast<bool>(octaveNum));

	/*bool isOctaveCalkNeed = octaveNum > 0 ? true : false;
	void *args[] = {&dev_noise, &dev_octave, &dev_k, &step, &numSteps, &isOctaveCalkNeed};
	cudaLaunchCooperativeKernel((void *)Perlin1D_kernel<T>, blocksPerGrid, threadsPerBlock, args, 0, 0);/**/
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": Perlin1D_kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\r\n";
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaDeviceSynchronize returned " << cudaGetErrorString(cudaStatus) << " after launching Perlin1D_kernel!\r\n";
		goto Error;
	}

	// Выполняем наложение октав на получившийся шум.
	if(octaveNum)
		if(resultDotsCols <= 2 * 32 * 1024 / sizeof(T)) // если вся октава помещается в разделяемую память, вызываем простое ядро
			Perlin1Doctave_shared_kernel<T> <<<blocksPerGrid, threadsPerBlock>>> (dev_noise, dev_octave, resultDotsCols, octaveNum);
		else // если октава не помещается целиком, вызываем сложное ядро.
			Perlin1Doctave_shared_unlimited_kernel<T> <<<blocksPerGrid, threadsPerBlock>>> (dev_noise, dev_octave, resultDotsCols, octaveNum);
		
	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": Perlin1Doctave_shared(_unlimited)_kernel launch failed: " << cudaGetErrorString(cudaStatus) << "\r\n";
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaDeviceSynchronize returned " << cudaGetErrorString(cudaStatus) << " after launching Perlin1Doctave_shared(_unlimited)_kernel!\r\n";
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(noise, dev_noise, resultDotsCols * sizeof(T), cudaMemcpyDeviceToHost);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": cudaMemcpy failed!\r\n";
		goto Error;
	}

Error:
	cudaFree(dev_noise);
	cudaFree(dev_octave);
	cudaFree(dev_k);

	return cudaStatus;
}