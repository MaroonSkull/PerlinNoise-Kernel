#pragma once

#include "Definitions.hpp"


class Window {
private:
	std::vector<GLuint> VAOs_;
	std::vector<GLuint> VBOs_;
	GLFWwindow *window = nullptr;
	std::string title_;

public:
	Window(int32_t width, int32_t height, const std::string &title, GLFWerrorfun errorCallback, GLFWframebuffersizefun resizeCallback);
	~Window();

	void reserveSimpleObject(size_t vectorSize = 1);
	size_t createVAO();
	size_t createVBO();
	void bindFillBuffer(size_t VAOId, size_t VBOId, std::vector<float> &data);

	GLFWwindow *getWindow() const;
	GLuint getVAO(size_t id = 0) const;

};