#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Интерфейс библиотеки
 */
class IPerlin {
public:
  /// Размерность пространства (1D, 2D, 3D...)
  std::size_t numberOfDimensions_{1};

  /// шаг между узлами градиентов [1, +inf)
  std::size_t pointsBetweenGradients_{1};

  /// количество векторов градиентов
  std::size_t numberOfGradients_{1};

  /// the number of levels of detail you want you perlin noise to have.
  std::size_t numberOfOctaves_{0};

  /// number that determines how much detail is added or removed at each octave
  /// (adjusts frequency)
  std::size_t lacunarity_{2};

  /// number that determines how much each octave contributes to the overall
  /// shape (adjusts amplitude).
  std::size_t persistence_{2};

  /**
   * @brief Длина одной оси
   * @details Количество опорных точек + количество промежутков между ними,
   * умноженное на длину одного промежутка
   */
  std::size_t getAxisLenght() {
    auto numberOfIntervals{numberOfGradients_ - 1};
    auto numberOfNodes{numberOfGradients_};
    return pointsBetweenGradients_ * numberOfIntervals + numberOfNodes;
  }

  float getDistanceBetweenTwoPoints() {
    return 1.0 / (pointsBetweenGradients_ + 1);
  }

  /**
   * @brief Конструктор класса Perlin с параметрами размерности и шага
   * градиента.
   *
   * @param N Размерность пространства шума.
   * @param gradientStep Шаг градиента для вычисления шума.
   */
  IPerlin(std::size_t numberOfDimensions, std::size_t pointsBetweenGradients,
          std::size_t numberOfGradients, std::size_t numberOfOctaves,
          std::size_t lacunarity, std::size_t persistence)
      : numberOfDimensions_{numberOfDimensions},
        pointsBetweenGradients_{pointsBetweenGradients},
        numberOfGradients_{numberOfGradients},
        numberOfOctaves_{numberOfOctaves}, lacunarity_{lacunarity},
        persistence_{persistence} {}

  IPerlin() = default;

  virtual ~IPerlin() = default;

  virtual std::vector<float> getNoise() = 0;
};

// forward declaration
class PerlinImpl;

class PimplPerlinFactory {
public:
  static std::unique_ptr<IPerlin> createPerlin();
};