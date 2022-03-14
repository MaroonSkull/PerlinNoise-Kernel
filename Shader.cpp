#include "Definitions.h"

bool Shader::readFromFile(std::string *accumulator, const GLchar *pathToFile) {
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

Shader::Shader(const char *vertexShaderPath, const char *fragmentShaderPath) {
	// Create vertex shader
	uint32_t vertexShader = glCreateShader(GL_VERTEX_SHADER);
	// Create fragment shader
	uint32_t fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	int32_t success;
	char infoLog[512];

	// Read vertex shader source code
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
		}
	}
	else std::cout << "ERROR::SHADER::VERTEX::MISSING_FILE\n" << std::endl;

	// Read vertex shader source code
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
			std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED" << std::endl << infoLog << std::endl;
		}
	}
	else std::cout << "ERROR::SHADER::FRAGMENT::MISSING_FILE" << std::endl;

	// Создаём объект шейдерной программы
	shaderId = glCreateProgram();

	// Прикрепляем наши шейдеры к шейдерной программе
	glAttachShader(shaderId, vertexShader);
	glAttachShader(shaderId, fragmentShader);
	glLinkProgram(shaderId);

	// Check shader program linking errors
	glGetProgramiv(shaderId, GL_LINK_STATUS, &success);
	if(!success) {
		glGetProgramInfoLog(shaderId, 512, nullptr, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED" << std::endl << infoLog << std::endl;
	}

	// Delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}

Shader::~Shader() {
	glDeleteProgram(shaderId);
}

void Shader::use() {
	glUseProgram(shaderId);
}

void Shader::setBool(const std::string &name, bool value) const {
	glUniform1i(glGetUniformLocation(shaderId, name.c_str()), static_cast<int32_t>(value));
}

void Shader::setInt(const std::string &name, int32_t value) const {
	glUniform1i(glGetUniformLocation(shaderId, name.c_str()), value);
}

void Shader::setFloat(const std::string &name, float value) const {
	glUniform1f(glGetUniformLocation(shaderId, name.c_str()), value);
}

int Shader::getShaderId() const {
    return shaderId;
}

void Shader::setShaderId(uint32_t shaderId) {
    this->shaderId = shaderId;
}