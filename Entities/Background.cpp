#include "../Services/Definitions.hpp"
#include "Background.hpp"

Background::Background(Window &ConcreteWindow, Params &p) : p_(p), IDrawable(ConcreteWindow) {}

void Background::draw() {
    glClearColor(p_.backgroundColor.x, p_.backgroundColor.y, p_.backgroundColor.z, p_.backgroundColor.w);
    glClear(GL_COLOR_BUFFER_BIT);
}
