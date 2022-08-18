/*
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <torch/script.h>
#include <vector>

inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
  std::vector<int64_t> result;
  result.reserve(list.size());
  for (size_t i = 0; i < list.size(); i++)
    result.push_back(list[i]);
  return result;
}

torch::Tensor broadcast(torch::Tensor src, torch::Tensor other, int64_t dim) {
  if (src.dim() == 1)
    for (auto i = 0; i < dim; i++)
      src = src.unsqueeze(0);
  for (auto i = src.dim(); i < other.dim(); i++)
    src = src.unsqueeze(-1);
  src = src.expand(other.sizes().vec());
  return src;
}
