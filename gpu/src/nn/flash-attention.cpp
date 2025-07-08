#include <pybind11/embed.h>
#include <memory>
#include <iostream>
#include "tensor.h"
#include "function.h"
namespace py = pybind11;

std::shared_ptr<Tensor> call_triton_attention(
    const std::shared_ptr<Tensor>& q,
    const std::shared_ptr<Tensor>& k,
    const std::shared_ptr<Tensor>& v,
    bool causal,
    float sm_scale,
    std::shared_ptr<Function> ctx
) {
    static py::scoped_interpreter guard{}; // 初始化 Python
    static py::object attention_fn = py::module_::import("your_module").attr("attention");

    try {
        py::object py_q = tensor_to_pytorch(q);
        py::object py_k = tensor_to_pytorch(k);
        py::object py_v = tensor_to_pytorch(v);

        py::object result = attention_fn(py_q, py_k, py_v, causal, sm_scale);
        std::shared_ptr<Tensor> out = pytorch_to_tensor(result);

        if ((q && q->requires_grad()) || (k && k->requires_grad()) || (v && v->requires_grad())) {
            if (ctx) {
                ctx->save({q, k, v, out});
                out->set_ctx(ctx);
            }
            out->_requires_grad = true;
        }

        return out;
    } catch (const py::error_already_set& e) {
        std::cerr << "Python error: " << e.what() << std::endl;
        return nullptr;
    }
}