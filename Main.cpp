#include "Definitions.h"
#include "Kernels.cu"

// Path to source of OpenGL vertex shader
const GLchar *vertexShaderPath = "Shader.vs";
// Path to source of fragment shader
const GLchar *fragmentShaderPath = "Shader.fs";

bool readFromFile(std::string *accumulator, const GLchar *pathToFile) {
	std::ifstream fileStream(pathToFile, std::ios::in);

	if(!fileStream.is_open()) {	// Если не вышло открыть файл, возвращаем ошибку
		std::cerr << "Could not read file " << pathToFile << ". File does not exist." << std::endl;
		return false;
	}

	std::string line = "";					// Создаём буфер
	while(!fileStream.eof()) {				// Читаем в него файл построчно
		std::getline(fileStream, line);
		accumulator->append(line + "\n");
	}

	fileStream.close();	// Подметаем за собой
	return true;
}

int main() {
	// Data in stack
	constexpr uint32_t controlPoints = 6;
	constexpr uint32_t numSteps = 4000;
	constexpr uint32_t octaveNum = 15;
	constexpr uint32_t resultDotsCols = (controlPoints - 1) * numSteps;
	constexpr float step = 1.0f / numSteps;
	constexpr float k[controlPoints] = {.6f, -.2f, 1.0f, -.6f, -.1f, .6f}; // значения наклонов на углах отрезков (последний наклон равен первому)
	// Perlin noise coords data in heap
	float *noise = new float[resultDotsCols+1];
	float *vertices = new float[3 * (resultDotsCols+1)]; //x, y, z to 1 dot -> length = 3*cols

	//for(int i = 0; i < resultDotsCols; i++)
		//noise[i] = 0.f;
	// Инициализируем z-координату графика 0
	for(int i = 0; i < resultDotsCols+1; i++)
		vertices[3 * i + 2] = 0.f;

	// Create OpenGL 3.3 context
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	GLFWwindow *window = glfwCreateWindow(1800, 600, "Perlin Noise Generator", nullptr, nullptr);
	if(window == nullptr) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Setting up viewport
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); // Устанавливаем callback на изменение размеров окна

	// Initialize GLAD
	if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -2;
	}

	// Calculate Perlin in parallel.
	cudaError_t cudaStatus = Perlin1DWithCuda<float>(noise, k, step, numSteps, controlPoints, resultDotsCols, octaveNum);
	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": Perlin1DWithCuda failed!\r\n";
		return -5;
	}

	{
		// Save dots into 3d coords
		for(int i = 0; i < resultDotsCols; i++) {
			vertices[3 * i] = 2 * static_cast<float>(i) / static_cast<float>(resultDotsCols) - 1; // x = 2x(norm)-1, нормализуем и смещаем влево
			vertices[3 * i + 1] = noise[i]; // y
			/*std::cout << "x[" << i << "] = " << vertices[3 * i] << "\t"
						<< "y[" << i << "] = " << vertices[3 * i + 1]	<< "\t"
						<< "z[" << i << "] = " << vertices[3 * i + 2]	<< "\r\n";/**/
		}
		vertices[3 * resultDotsCols] = 1; // Последняя точка всегда равна нулю.
		vertices[3 * resultDotsCols + 1] = 0;

		// Create vertex array object.
		uint32_t VAO;
		glGenVertexArrays(1, &VAO);
		std::cout << "Vertex array object have been created with ID = " << VAO << "\r\n";

		// Связываем объект вершинного массива.
		glBindVertexArray(VAO);

		// Create vertex buffer object.
		uint32_t VBO;
		glGenBuffers(1, &VBO);
		std::cout << "Vertex buffer object have been created with ID = " << VBO << "\r\n";

		// Связываем буфер. Теперь все вызовы буфера с параметром GL_ARRAY_BUFFER
		// будут использоваться для конфигурирования созданного буфера VBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		// Копируем данные вершин в память связанного буфера
		glBufferData(GL_ARRAY_BUFFER, 3 * (resultDotsCols+1) * sizeof(*vertices), vertices, GL_STATIC_DRAW);

		// Сообщаем, как OpenGL должен интерпретировать данные вершин,
		// которые мы храним в vertices[]
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
		glEnableVertexAttribArray(0);

		// Create vertex shader
		uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);

		// Create fragment shader
		uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

		int32_t success;
		char infoLog[512];

		// Read vertex shader source code
		{ // Изолируем временные переменные
			std::string vs;
			if(readFromFile(&vs, vertexShaderPath)) {
				const char *vertexShaderSource = vs.c_str();
				// Compile vertex shader source code
				glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
				glCompileShader(vertexShader);
				// Check vertex shader compile errors
				glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
				if(!success) {
					glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
					std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
					return -3;
				}
				else std::cout << "Vertex shader have been compiled!\r\n";
			}
			else return -10;
		}

		// Read vertex shader source code
		{ // Изолируем временные переменные
			std::string fs;
			if(readFromFile(&fs, fragmentShaderPath)) {
				const char *fragmentShaderSource = fs.c_str();
				// compile fragment shader
				glShaderSource(fragmentShader, 1, &fragmentShaderSource, nullptr);
				glCompileShader(fragmentShader);
				// Check fragment shader compile errors
				glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
				if(!success) {
					glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
					std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\r\n" << infoLog << "\r\n\r\n"
						<< "Source code:\r\n" << fragmentShaderSource << "/end/\r\n";

					return -4;
				}
				else std::cout << "Fragment shader have been compiled!\r\n";
			}
			else return -11;
		}

		// Создаём объект шейдерной программы
		uint32_t shaderProgram = glCreateProgram();

		// Прикрепляем наши шейдеры к шейдерной программе
		glAttachShader(shaderProgram, vertexShader);
		glAttachShader(shaderProgram, fragmentShader);
		glLinkProgram(shaderProgram);

		// Check shader program linking errors
		glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
		if(!success) {
			glGetProgramInfoLog(shaderProgram, 512, nullptr, infoLog);
			std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
			return -6;
		}
		else std::cout << "Shader program have been linked!\r\n";

		// Delete the shaders as they're linked into our program now and no longer necessery
		glDeleteShader(vertexShader);
		glDeleteShader(fragmentShader);

		// Create render cycle
		while(!glfwWindowShouldClose(window)) {
			// Input processing
			processInput(window);

			// Rendering
			// Активируем созданный объект
			glUseProgram(shaderProgram);

			// Отменяем связывание???
			glBindVertexArray(VAO);

			// Рисуем ось OX


			// Рисуем шум Перлина
			glDrawArrays(GL_LINE_STRIP, 0, resultDotsCols+1);

			// Swap buffers
			glfwSwapBuffers(window);
			glfwPollEvents();
		}
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();
	// glfwTerminate must be called before exiting in order for clean up
	glfwTerminate();
	return 0;
}

// Обработка ресайза окна
void framebuffer_size_callback(GLFWwindow *window, int32_t width, int32_t height) {
	glViewport(0, 0, width, height);
}

// Обработка всех событий ввода: запрос GLFW о нажатии/отпускании клавиш на клавиатуре в данном кадре и соответствующая обработка данных событий
void processInput(GLFWwindow *window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}