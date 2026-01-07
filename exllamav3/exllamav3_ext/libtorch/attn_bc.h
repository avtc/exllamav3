#pragma once

#include "attn.h"
#include <pybind11/pybind11.h>
namespace py = pybind11;

py::class_<BC_Attention, std::shared_ptr<BC_Attention>>(m, "BC_Attention")
    .def(
        py::init<
            std::shared_ptr<BC_LinearEXL3>, // q_proj
            std::shared_ptr<BC_LinearEXL3>, // k_proj
            std::shared_ptr<BC_LinearEXL3>, // v_proj
            std::shared_ptr<BC_LinearEXL3>, // o_proj
            at::Tensor, // q_norm_weight
            at::Tensor, // k_norm_weight
            float,      // norm_epsilon
            int,        // hidden_size
            int,        // num_heads
            int,        // head_dim
            int         // num_kv_heads
        >(),
        py::arg("q_proj"),
        py::arg("k_proj"),
        py::arg("v_proj"),
        py::arg("o_proj"),
        py::arg("q_norm_weight"),
        py::arg("k_norm_weight"),
        py::arg("norm_epsilon"),
        py::arg("hidden_size"),
        py::arg("num_heads"),
        py::arg("head_dim"),
        py::arg("num_kv_heads")
    )
    .def("run_proj", &BC_Attention::run_proj)
    .def("run_out", &BC_Attention::run_out);
