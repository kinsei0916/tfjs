/* Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ===========================================================================*/

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#include <cmath>
#include <cstddef>

#include "tfjs-backend-wasm/src/cc/binary.h"
#include "tfjs-backend-wasm/src/cc/util.h"

namespace {

template <typename T>
inline T mod(T a, T b) {
  struct FloatMod {
    float operator()(const float lhs, const float rhs) const {
      return std::fmod(lhs, rhs);
    }
  };
  using ModFunc = typename std::conditional<std::is_integral<T>::value,
                                            std::modulus<T>, FloatMod>::type;
  ModFunc mod_func;
  T trunc_mod = mod_func(a, b);
  return (trunc_mod != 0) && ((b < 0) != (trunc_mod < 0)) ? (trunc_mod + b)
                                                          : trunc_mod;
}

}  // namespace

namespace tfjs {
namespace wasm {
// We use C-style API to interface with Javascript.
extern "C" {

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void Mod(const int a_id, const size_t* a_shape_ptr, const int a_shape_len,
         const int b_id, const size_t* b_shape_ptr, const int b_shape_len,
         const DType dtype, const int out_id) {
  switch (dtype) {
    case DType::float32:
      binary_f32(a_id, b_id, out_id, mod);
      break;
    case DType::int32:
      binary_i32(a_id, b_id, out_id, mod);
      break;
    default:
      util::warn("Mod for tensor ids %d and %d failed. Unsupported dtype %d",
                 a_id, b_id, dtype);
  }
}

}  // extern "C"
}  // namespace wasm
}  // namespace tfjs
