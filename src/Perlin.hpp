#include <cstddef>
#include <memory>
#include <vector>

/**
 * @brief Интерфейс библиотеки
 */
class IPerlin {
public:
  /// Размерность пространства (1D, 2D, 3D...)
  int numberOfDimensions_{1};

  /// шаг между узлами градиентов [1, +inf)
  int pointsBetweenGradients_{1};

  /// количество векторов градиентов
  int numberOfGradients_{1};

  /// the number of levels of detail you want you perlin noise to have.
  int numberOfOctaves_{0};

  /// number that determines how much detail is added or removed at each octave
  /// (adjusts frequency)
  float lacunarity_{2};

  /// number that determines how much each octave contributes to the overall
  /// shape (adjusts amplitude).
  float persistence_{2};

  /**
   * @brief Длина одной оси
   * @details Количество опорных точек + количество промежутков между ними,
   * умноженное на длину одного промежутка
   */
  int getAxisLenght() {
    auto numberOfIntervals{numberOfGradients_ - 1};
    auto numberOfNodes{numberOfGradients_};
    return pointsBetweenGradients_ * numberOfIntervals + numberOfNodes;
  }

  float getDistanceBetweenTwoPoints() {
    return 1.0 / (pointsBetweenGradients_ + 1);
  }

  /**
   * @brief Конструктор класса Perlin.
   */
  IPerlin() = default;

  virtual ~IPerlin() = default;

  virtual std::vector<float> &initializeGradients() = 0;
  virtual std::vector<float> &getNoise() = 0;
};

class PimplPerlinFactory {
public:
  static std::unique_ptr<IPerlin> createPerlin();
};