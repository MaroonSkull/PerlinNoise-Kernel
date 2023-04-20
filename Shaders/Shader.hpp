#pragma once
#include <array>


class Shader {
private:
    bool readFromFile(std::string *accumulator, std::string_view pathToFile);
    // the program ID
    size_t shaderId_ = 0;

public:
    // constructor reads and builds the shader
    Shader(std::string_view vertexShaderPath, std::string_view fragmentShaderPath);
    Shader();
    ~Shader();

    // use/activate the shader
    void use() const;

    void set(const std::string &name, GLfloat value) const;
    void set(const std::string &name, const std::array<GLfloat, 4> &value) const;
    void set(const std::string &name, bool value) const;
};