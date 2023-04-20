// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com

#include "Services/Definitions.hpp"

std::mt19937 gen{std::random_device{}()};
std::normal_distribution vectorGenerator{.0, .5}; //95.45% значений внутри [-1, 1]
std::uniform_int_distribution eventlyGen{2, 15};
std::uniform_int_distribution dotsGen{10, 50};

// todo сделать локальной, разбить на глобальное состояние и на локальные мелкие
Params p{[&]() -> float { return std::clamp(vectorGenerator(gen), -1., 1.); },
		static_cast<uint32_t>(eventlyGen(gen)), static_cast<uint32_t>(dotsGen(gen)), static_cast<uint32_t>(eventlyGen(gen))};



/*Params p1{[&]() -> float { return std::clamp(vectorGenerator(gen), -1., 1.); },
		static_cast<uint32_t>(eventlyGen(gen)), static_cast<uint32_t>(dotsGen(gen)), static_cast<uint32_t>(eventlyGen(gen))};
*/

int main() {
	try {
	
		Window MainWindow(1800, 600, "Perlin Noise Generator", GLFWErrorCallback, framebufferSizeCallback);

		/*Params p{
			[&]() -> float {
				return std::clamp(vectorGenerator(gen), -1., 1.);
			},
			static_cast<uint32_t>(eventlyGen(gen)),
			static_cast<uint32_t>(dotsGen(gen)),
			static_cast<uint32_t>(eventlyGen(gen))
		};*/

		PerlinNoise(MainWindow, "shaders/1D/1D_noise.vert", "shaders/1D/1D_white.vert", "shaders/1D/1D.frag");
	
	}
	catch(const cudaWrp::ErrorException &e) {
		std::cerr << e.getError();
	}
	catch(const std::exception &e) {
		std::cerr << "Standard exception: " << e.what() << std::endl;
	}
	catch(const char *e) {
		std::cerr << e << std::endl;
	}
	catch(...) {
		std::cerr << "Unknown exception." << std::endl;
	}

	glfwTerminate();
	cudaWrp::destroyContext();
	
	return 0;
}


int PerlinNoise(Window &MainWindow, std::string_view vertexShaderPathNoise, std::string_view vertexShaderPathLinear, std::string_view fragmentShaderPath) {

	Shader PlotShader(vertexShaderPathLinear, fragmentShaderPath);
	Shader PerlinShader(vertexShaderPathNoise, fragmentShaderPath);

	Background BackgroundLayer(MainWindow, p);
	// todo разбить Plot на сетку и на класс для угловых наклонов
	Plot PlotLayer(MainWindow, PlotShader, p);
	//Plot PlotLayer1(MainWindow, PlotShader, p1);
	Perlin PerlinLayer(MainWindow, PerlinShader, p);
	//Perlin PerlinLayer1(MainWindow, PerlinShader, p1);
	GUI GUILayer(MainWindow, p);

	glfwSetScrollCallback(MainWindow.getWindow(), scrollCallback);
	glfwSetKeyCallback(MainWindow.getWindow(), keyCallback);
	// Create render cycle
	while(!glfwWindowShouldClose(MainWindow.getWindow())) {
		// Input processing
		processInput(MainWindow.getWindow());

		// Calculable
		PlotLayer.calc();
		//PlotLayer1.calc();
		PerlinLayer.calc();
		//PerlinLayer1.calc();
		GUILayer.calc();

		// Drawing
		BackgroundLayer.draw();
		PlotLayer.draw();
		//PlotLayer1.draw();
		PerlinLayer.draw();
		//PerlinLayer1.draw();
		GUILayer.draw();

		// Swap buffers
		glfwSwapBuffers(MainWindow.getWindow());
		glfwPollEvents();
	}

	return 0;
}

// Обработка любых ошибок GLFW
void GLFWErrorCallback(int error, const char *description) {
	std::cerr << stderr << std::endl << "Glfw Error " << error << ", " << description << std::endl;
}

// Обработка ресайза окна
void framebufferSizeCallback(GLFWwindow *window, int32_t width, int32_t height) {
	glViewport(0, 0, width, height);
}

// Обработка всех событий ввода: запрос GLFW о нажатии/отпускании клавиш
// на клавиатуре в данном кадре и соответствующая обработка данных событий
void processInput(GLFWwindow *window) {
	if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
}

// изменение угла наклона активной прямой
void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
	p.k_.at(p.activeId_) -= yoffset;
	p.k_.at(p.activeId_) = std::clamp(p.k_.at(p.activeId_), -1.f, 1.f);

	if(p.activeId_ == 0)
		p.k_.back() = p.k_.at(0);
	else if(p.activeId_ == p.k_.size() - 1)
		p.k_.at(0) = p.k_.back();
}

void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods) {
	// выбор активной прямой
	if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS) {
		p.incrActive();
	}
	if(key == GLFW_KEY_LEFT && action == GLFW_PRESS) {
		p.decrActive();
	}

	// регулировка количества опорных точек (прямых)
	if(key == GLFW_KEY_UP && action == GLFW_PRESS) {
		p.incrControlPoint();
	}
	if(key == GLFW_KEY_DOWN && action == GLFW_PRESS) {
		p.decrControlPoint();
	}

	// регулировка количеества наложений октав
	if(key == GLFW_KEY_W && action == GLFW_PRESS) {
		p.incrOctave();
	}
	if(key == GLFW_KEY_S && action == GLFW_PRESS) {
		p.decrOctave();
	}

	// регулировка количества точек на один промежуток
	if(key == GLFW_KEY_D && action == GLFW_PRESS) {
		p.incrNumSteps();
	}
	if(key == GLFW_KEY_A && action == GLFW_PRESS) {
		p.decrNumSteps();
	}

	// регулировка скорости прироста количества точек
	if(key == GLFW_KEY_LEFT_CONTROL) {
		if(action == GLFW_PRESS)
			p.dx_ = 10;
		else if(action == GLFW_RELEASE)
			p.dx_ = 1;
	}
	if(key == GLFW_KEY_LEFT_SHIFT) {
		if(action == GLFW_PRESS)
			p.dx_ = 100;
		else if(action == GLFW_RELEASE)
			p.dx_ = 1;
	}
}
