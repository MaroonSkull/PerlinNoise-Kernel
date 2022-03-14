#include "Definitions.h"

int main() {
	//return Perlin1D<float>("vertexShader1D_noise.vs", "fragmentShader1D.fs");
	return Perlin2D<float>("vertexShader2D.vs", "fragmentShader2D.fs");
}

template <typename T>
int Perlin2D(const char *vertexShaderPath, const char *fragmentShaderPath) {
	// Create OpenGL 3.3 context
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	GLFWwindow *window = glfwCreateWindow(640, 480, "Perlin Noise Generator", nullptr, nullptr);
	if(window == nullptr) {
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	// Setting up viewport
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback_texture_edition); // Устанавливаем callback на изменение размеров окна

	// Initialize GLAD
	if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cout << "Failed to initialize GLAD" << std::endl;
		return -2;
	}

	Shader shader(vertexShaderPath, fragmentShaderPath);

	float verticess[] = {
		-0.5,  0.5, 0.0,	// top left
		 0.5,  0.5, 0.0,	// top right
		-0.5, -0.5, 0.0,	// bottom left
		 0.5, -0.5, 0.0,	// bottom right
	};

	uint32_t indices[] = {
		0, 1, 2,
		1, 2, 3
	};

	uint32_t VAO, VBO, EBO, texture;
	glGenTextures(1, &texture);

	// Create vertex array object.
	glGenVertexArrays(1, &VAO);
	std::cout << "Vertex array object have been created with ID = " << VAO << "\r\n";

	// Create vertex buffer object.
	glGenBuffers(1, &VBO);
	std::cout << "Vertex buffer object have been created with ID = " << VBO << "\r\n";

	// Create element buffer object.
	glGenBuffers(1, &EBO);
	std::cout << "element buffer object have been created with ID = " << EBO << "\r\n";

	// Связываем объект массива вершин.
	glBindVertexArray(VAO);

	// Связываем буфер. Теперь все вызовы буфера с параметром GL_ARRAY_BUFFER
	// будут использоваться для конфигурирования созданного буфера VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	// Копируем данные вершин в память связанного буфера
	glBufferData(GL_ARRAY_BUFFER, sizeof(verticess), verticess, GL_STATIC_DRAW);

	// Пробуем биндить объект буфера эллементов
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// Сообщаем, как OpenGL должен интерпретировать данные вершин,
	// которые мы храним в verticess[]
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);
	glEnableVertexAttribArray(0);

	// note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
	// VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
	glBindVertexArray(0);

	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	// Create render cycle
	while(!glfwWindowShouldClose(window)) {
		// Input processing
		processInput(window);

		// Rendering
		// Активируем созданный объект
		shader.use();

		// Применяем всё, что применяли до этого
		glBindVertexArray(VAO);
		// Рисуем свои треугольники
		glDrawElements(GL_TRIANGLES, sizeof(indices) / sizeof(indices[0]), GL_UNSIGNED_INT, 0);
		glBindVertexArray(0);

		// Swap buffers
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaDeviceReset();

	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();
	return 0;
}

template <typename T>
int Perlin1D(const char *vertexShaderPathNoise, const char *fragmentShaderPath) {
	// Data in stack
	const char *vertexShaderPathLinear = "vertexShader1D_linear.vs";
	cudaError_t cudaStatus = cudaError::cudaErrorUnknown;
	constexpr uint32_t controlPoints = 7;
	constexpr uint32_t numSteps = 2000;
	constexpr uint32_t octaveNum = 8;
	constexpr uint32_t resultDotsCols = (controlPoints - 1) * numSteps;
	constexpr T step = 1.0f / numSteps;
	constexpr T k[controlPoints] = {.6f, -.2f, 1.0f, -.6f, -.1f, .5f, .6f/**/}; // значения наклонов на углах отрезков (последний наклон равен первому)
	// Perlin noise coords data in heap
	T *noise = new T[resultDotsCols + 1];
	T *vertices = nullptr;

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

	vertices = new T[2 * (resultDotsCols + 1)]; //x, y to 1 dot -> length = 2*cols

	// Calculate Perlin in parallel.
	if constexpr(std::is_same<T, float>::value)
		cudaStatus = Perlin1DWithCuda_f(noise, k, step, numSteps, controlPoints, resultDotsCols, octaveNum);
	else
		cudaStatus = Perlin1DWithCuda_d(noise, k, step, numSteps, controlPoints, resultDotsCols, octaveNum);

	if(cudaStatus != cudaSuccess) {
		std::cout << stderr << ": Perlin1DWithCuda failed!\r\n";
		return -5;
	}

	{
		// Save dots into 3d coords
		for(int i = 0; i < resultDotsCols; i++) {
			vertices[2 * i] = 2 * static_cast<T>(i) / resultDotsCols - 1; // x = 2x(norm)-1, нормализуем и смещаем влево
			vertices[2 * i + 1] = noise[i]; // y
		}
		vertices[2 * resultDotsCols] = 1.0f; // Последняя точка всегда равна нулю.
		vertices[2 * resultDotsCols + 1] = 0.0f;

		float ks[8 + 2 * 2 * controlPoints] = {
			-1.0, 0,
			1.0, 0,
			0, -1,
			0, 1,
		}; // x, y на каждую из 2*controlPoints точек для прямых + 8 точек для осей OXY.
		for(int i = 0; i < controlPoints; i++) {
			float x0 = static_cast<T>(i) / (controlPoints - 1);
			float deltaX = static_cast<T>(1) / (controlPoints - 1)/4;

			ks[8 + 4 * i] = 2 * (x0 - deltaX) - 1;						// x left
			ks[8 + 4 * i + 1] = (controlPoints - 1) * k[i] * -deltaX;	// y left
			ks[8 + 4 * i + 2] = 2 * (x0 + deltaX) - 1;					// x right
			ks[8 + 4 * i + 3] = (controlPoints - 1) * k[i] * deltaX;	// y right
		}

		// Create vertex array object.
		uint32_t VAO, VAO1, VBO, VBO1;
		glGenVertexArrays(1, &VAO);
		std::cout << "Vertex array object have been created with ID = " << VAO << "\r\n";
		glGenVertexArrays(1, &VAO1);
		std::cout << "Vertex array object have been created with ID = " << VAO1 << "\r\n";
		// Create vertex buffer object.
		glGenBuffers(1, &VBO);
		std::cout << "Vertex buffer object have been created with ID = " << VBO << "\r\n";
		glGenBuffers(1, &VBO1);
		std::cout << "Vertex buffer object have been created with ID = " << VBO1 << "\r\n";

		// Связываем объект вершинного массива.
		glBindVertexArray(VAO);

		// Связываем буфер. Теперь все вызовы буфера с параметром GL_ARRAY_BUFFER
		// будут использоваться для конфигурирования созданного буфера VBO
		glBindBuffer(GL_ARRAY_BUFFER, VBO);

		// Копируем данные вершин в память связанного буфера
		glBufferData(GL_ARRAY_BUFFER, 2 * (resultDotsCols + 1) * sizeof(*vertices), vertices, GL_STATIC_DRAW);

		// Сообщаем, как OpenGL должен интерпретировать данные вершин,
		// которые мы храним в vertices[]
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
		glEnableVertexAttribArray(0);

		// Связываем объект вершинного массива.
		glBindVertexArray(VAO1);

		glBindBuffer(GL_ARRAY_BUFFER, VBO1);
		glBufferData(GL_ARRAY_BUFFER, sizeof(ks), ks, GL_STATIC_DRAW);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
		glEnableVertexAttribArray(0);

		Shader shader(vertexShaderPathNoise, fragmentShaderPath);
		Shader shaderSolid(vertexShaderPathLinear, fragmentShaderPath);

		// Create render cycle
		while(!glfwWindowShouldClose(window)) {
			// Input processing
			processInput(window);

			// Rendering

			// Активируем созданную программу
			shaderSolid.use();
			glLineWidth(1.0);
			glBindVertexArray(VAO1);
			for(int i = 0; i < controlPoints + 4; i++) {
				glDrawArrays(GL_LINE_STRIP, 2 * i, 2);
			}

			// Активируем созданную программу
			shader.use();
			glLineWidth(1.1);
			// предоставляем выполнение бинда gl
			glBindVertexArray(VAO);

			// Рисуем шум Перлина
			glDrawArrays(GL_LINE_STRIP, 0, resultDotsCols + 1);

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

// Обработка ресайза окна для 2D
void framebuffer_size_callback_texture_edition(GLFWwindow *window, int32_t width, int32_t height) {
	//тут надо пересоздавать текстуру и перепривязывать её к cuda
	glViewport(0, 0, width, height);
}

// Обработка всех событий ввода: запрос GLFW о нажатии/отпускании клавиш на клавиатуре в данном кадре и соответствующая обработка данных событий
void processInput(GLFWwindow *window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}