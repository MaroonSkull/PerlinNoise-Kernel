#pragma once

#include "Params.hpp"

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
template<typename T> void Perlin1D(T *vertices, const Params1D &p);