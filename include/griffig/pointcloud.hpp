#pragma once

#include <vector>

#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <movex/affine.hpp>


/* struct Pointcloud {
    struct PointXYZRGB {
        float x, y, z;
        float r, g, b;
    };

    size_t width, height; // For image-based pointclouds

    std::vector<PointXYZRGB> points;
    movex::Affine pose;  // Sensor pose

    Pointcloud() { }
    Pointcloud(size_t width, size_t height): width(width), height(height) {
        resize(width * height);
    }

    void resize(size_t count) {
        points.resize(count);
        if (count != width * height) {
            width = count;
            height = 1;
        }
    }
}; */


struct Vertex {
    float x, y, z;

    operator const float* () const {
        return &x;
    }
};


struct TexCoord {
    float u, v;

    operator const float* () const {
        return &u;
    }
};


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
    size_t count;
    const void* vertices;
    const Texture& tex;
    const void* tex_coords;

    explicit Pointcloud(size_t count, const void* vertices, const Texture& tex, const void* tex_coords): tex(tex), count(count), vertices(vertices), tex_coords(tex_coords) { }
};
