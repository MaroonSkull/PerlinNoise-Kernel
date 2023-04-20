#include "Definitions.hpp"
#include "Window.hpp"



Window::Window(int32_t width, int32_t height, const std::string &title, GLFWerrorfun errorCallback, GLFWframebuffersizefun resizeCallback)
	: title_(title) {
	
	glfwSetErrorCallback(errorCallback);

	// Create OpenGL 3.3 context
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create window
	window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
	if(window == nullptr) {
		glfwTerminate();
		throw "Failed to create GLFW window";
	}
	glfwMakeContextCurrent(window);
	//glfwSwapInterval(1); // Enable vsync

	// Setting up viewport
	glfwSetFramebufferSizeCallback(window, resizeCallback); // Устанавливаем callback на изменение размеров окна

	// Initialize GLAD
	if(!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		throw "Failed to initialize GLAD";
	}
}

Window::~Window() {
	if(!window) glfwDestroyWindow(window);
}

// функция работает, предполагая, что каждому vao соответствует свой vbo
size_t Window::createVAO() {
	// Create vertex array object.
	VAOs_.push_back(NAN);
	GLuint *VAO = &VAOs_.back();
	glGenVertexArrays(1, VAO);
	std::cout << "Vertex array object have been created with ID = " << *VAO << "\r\n";
	return VAOs_.size() - 1;
}

size_t Window::createVBO() {
	// Create vertex buffer object.
	VBOs_.push_back(NAN);
	GLuint *VBO = &VBOs_.back();
	glGenBuffers(1, VBO);
	std::cout << "Vertex buffer object have been created with ID = " << *VBO << "\r\n";

	return VBOs_.size() - 1;
}

void Window::bindFillBuffer(size_t VAOId, size_t VBOId, std::vector<float> &data) {
	// Связываем объект вершинного массива.
	glBindVertexArray(VAOs_.at(VAOId));

	// Связываем буфер. Теперь все вызовы буфера с параметром GL_ARRAY_BUFFER
	// будут использоваться для конфигурирования созданного буфера VBO
	glBindBuffer(GL_ARRAY_BUFFER, VBOs_.at(VBOId));

	// Копируем данные вершин в память связанного буфера
	glBufferData(GL_ARRAY_BUFFER, data.size()*sizeof(float), data.data(), GL_DYNAMIC_DRAW);
	
	// Сообщаем, как OpenGL должен интерпретировать данные вершин,
	// которые мы храним в vertices[]
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);
	
	glEnableVertexAttribArray(0);
	
}

void Window::reserveSimpleObject(size_t vectorSize) {
	if(VAOs_.capacity() < vectorSize) VAOs_.reserve(vectorSize);
	if(VBOs_.capacity() < vectorSize) VBOs_.reserve(vectorSize);
}

GLFWwindow *Window::getWindow() const {
	return window;
}

GLuint Window::getVAO(size_t id) const {
	return VAOs_.at(id);
}