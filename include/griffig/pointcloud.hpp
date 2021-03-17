#pragma once

#include <vector>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <movex/affine.hpp>


namespace PointTypes {

struct XYZ {
    float x, y, z;

    operator const float* () const {
        return &x;
    }
};

struct XYZRGB {
    float x, y, z;
    float r, g, b;

    operator const float* () const {
        return &x;
    }
};

struct UV {
    float u, v;

    operator const float* () const {
        return &u;
    }
};

} // namespace PointTypes


class Texture {
    GLuint gl_handle {0};

public:
    void upload(size_t width, size_t height, const void* data) {
        if (!gl_handle) {
            glGenTextures(1, &gl_handle);
        }

        glBindTexture(GL_TEXTURE_2D, gl_handle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    GLuint get_gl_handle() const {
        return gl_handle;
    }
};


struct Pointcloud {
    size_t size;
    size_t width, height;

    const void* vertices;
    const Texture& tex;
    const void* tex_coords;

    explicit Pointcloud(size_t size, const void* vertices, const Texture& tex, const void* tex_coords): tex(tex), size(size), vertices(vertices), tex_coords(tex_coords) { }
    // explicit Pointcloud(size_t width, size_t height, size_t size, const void* vertices, const void* texture, const void* tex_coords) {

    // }
};
