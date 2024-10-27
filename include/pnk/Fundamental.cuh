#pragma once
// ****************** Device functions ******************



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
__device__ __forceinline__ T
lerp_kernel(T k0, T k1, T t) {
	return fma(t, k1 - k0, k0); // (1-t)*k0 + t*k1 = k0 - t*k0 + t*k1 = t*(k1 - k0) + k0
}

/**
* Сигмоидальная функция из семейства smoothstep, используется для создания более интенсивного градиента шума.
* Подробнее см. https://en.wikipedia.org/wiki/Smoothstep#Variations
*
* \param x – значение градиента (он же t)
*
* \return возвращает классический smoothstep(x). Используется оригинальный первый полином.
*/
template <typename T>
__device__ __forceinline__ T
smoothstep_kernel(T x) {
	return fma(static_cast<T>(2), x, static_cast<T>(-3)) * -x * x; // 3 * x^2 - 2 * x^3 = -x * x * (2 * x - 3);
}

/**
* Сигмоидальная функция из семейства smoothstep, используется для создания ещё более интенсивного градиента шума.
* Подробнее см. https://en.wikipedia.org/wiki/Smoothstep#Variations
*
* \param x – значение градиента (он же t)
*
* \return возвращает классический smootherstep(x). Используется оригинальный второй полином Кена Перлина.
*/
template <typename T>
__device__ __forceinline__ T
smootherstep_kernel(T x) {
	return fma(static_cast<T>(6), x * x, fma(static_cast<T>(-15), x, static_cast<T>(10))) * x * x * x; // 6x^5 - 15x^4 + 10x^3 = x^3(6x^2 - 15x + 10)
}