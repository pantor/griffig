#pragma once

#include <affx/affine.hpp>


enum class ActionType {
    Grasp,
};


struct Grasp {
    using Affine = affx::Affine;

    ActionType action_type {ActionType::Grasp};

    Affine pose;
    double stroke;

    size_t index; // Index of inference model
    double estimated_reward;

    // explicit Grasp() { }
    explicit Grasp(const Affine& pose = Affine(), double stroke = 0.0, size_t index = 0, double estimated_reward = 0.0): pose(pose), stroke(stroke), index(index), estimated_reward(estimated_reward) { }

    std::string toString() const {
        return pose.toString() + " d: " + std::to_string(stroke) + " estimated reward: " + std::to_string(estimated_reward);
    }
};
