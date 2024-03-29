﻿#pragma once
// ****************** Global functions ******************



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
__global__ void Perlin1D_kernel(T *noise, T *octave, const T *k, uint32_t size, T step, uint32_t numSteps, bool isOctaveCalkNeed) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;// [0..] – всего точек для просчёта
	if(id >= size) return;
	uint32_t n = id * step;					// 0 0 0 / 1 1 1 / 2 2 2 / .. – какие точки к каким контрольным точкам принадлежат
	uint32_t dotNum = id % numSteps;		// 0 1 2 / 0 1 2 / 0 1 2 / .. – позиция точки между левым и правым значением
	T t = dotNum * step;					// 0.0 0.33 0.66 / 0 0.33 0.66 / .. – численное значение точки для интерполяции
	t = smoothstep_kernel<T>(t);			// Применяем сигмоидальную(на промежутке [0, 1]) функцию, реализуя градиент
	T y0 = k[n] * t;						// kx+b (b = 0)
	T y1 = k[n + 1] * (t - 1);				// kx+b (b = -k) = k(x-1)
	noise[id] = lerp_kernel<T>(y0, y1, t);	// Интерполяцией находим шум, пишем сразу в выходной массив

	// Если нужно вычислять октавы, сохраняем в памяти первую окатву шума
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
__global__ void Perlin1Doctave_shared_kernel(T *noise, const T *octave, uint32_t size, uint32_t octaveNum) {
	// выделяем разделяемую память для октав.
	/* используем 32KB памяти, на всех более-менее современных архитектурах (CC 3.7+)
	* именно такое значение позволит запускать минимум 2 блока на одном sm.
	* Это приведёт к потенциальной 100% занятости устройства.
	* Так же это даёт 8192 fp32 значения, или 4096 fp64. */
	constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
	__shared__ T sharedOctave[sharedOctaveLength];

	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	//if(id >= size) return; // здесь не делаем, потому что неиспользуемые сейчас потоки могут пригодиться в обработке других блоков

	if(size > 2 * sharedOctaveLength) {	// проверка на неправильный вызов ядра
		if(id == 0) printf("size = %d, 2*sharedOctaveLength = %d. exit.\r\n\r\n", size, 2 * sharedOctaveLength);
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
	for(uint32_t i = 0; i < (gridDim.x + 1) >> 1; i++) {
		uint32_t sharedId = blockDim.x * i + threadIdx.x;
		if(size > 2 * sharedId) // контроллируем выход за пределы массива
			sharedOctave[sharedId] = octave[sharedId];
	}

	// Синхронизируем выполнение на уровне блока.
	__syncthreads();
	// На этом моменте вся первая октава записана в разделяемую память данного блока

	// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
	for(int j = 1; j <= octaveNum; j++) {
		int octavePov = 1 << j;
#pragma unroll
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
__global__ void Perlin1Doctave_shared_unlimited_kernel(T *noise, const T *octave, uint32_t size, uint32_t octaveNum) {
	// выделяем разделяемую память для октав.
	/* используем 32KB памяти, на всех более-менее современных архитектурах (CC 3.7+)
	* именно такое значение позволит запускать минимум 2 блока на одном sm.
	* Это приведёт к потенциальной 100% занятости устройства.
	* Так же это даёт 8192 fp32 значения, или 4096 fp64. */
	constexpr uint32_t sharedOctaveLength = 32 * 1024 / sizeof(T);
	__shared__ T sharedOctave[sharedOctaveLength];

	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id >= size) return;

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

		// Применяем наложение октав, каждый раз основываясь на предыдущей октаве
		for(int j = 1; j <= octaveNum; j++) {
			int octavePov = 1 << j;
#pragma unroll
			for(int k = 0; k < octavePov; k++) {
				uint32_t globalMin = k * size / octavePov;
				uint32_t globalMax = (k + 1) * size / octavePov;
				uint32_t sharedMin = sharedOctaveLength / (octavePov / 2) * i + globalMin;
				uint32_t sharedMax = sharedOctaveLength / (octavePov / 2) * (i + 1) + globalMin;
				int32_t sharedId = id - sharedMin;
				if((id >= globalMin) && (id < globalMax) && (id >= sharedMin) && (id < sharedMax)) {
					if(sharedId * (octavePov >> 1) >= sharedOctaveLength) printf("outOfRangeAdress! k = %d, id = %d, sharedId = %d, octavePov>>1 = %d\r\n", k, id, sharedId, octavePov >> 1);
					else {
						noise[id] += sharedOctave[sharedId * (octavePov >> 1)] / (octavePov >> 1);
						break;
					}
				}
			}
		}
	}
}

template <typename T>
__global__ void Perlin1Dvertices_kernel(T *vertices, const T *noise, uint32_t size) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= size) return;

	vertices[2 * id] = 2.0 / size * static_cast<float>(id) - 1.0;
	vertices[2 * id + 1] = noise[id];
	if(id == size - 1) {
		vertices[2 * id + 2] = 1.;
		vertices[2 * id + 3] = 0.;
	}
}