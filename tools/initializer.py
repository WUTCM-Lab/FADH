# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Parameter init."""
import math
import mindspore as ms
from mindspore import nn


def default_recurisive_init(custom_cell):
    """Initialize parameter."""
    for _, cell in custom_cell.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            cell.weight.set_data(ms.common.initializer.initializer(ms.common.initializer.HeUniform(math.sqrt(5)),
                                                                   cell.weight.shape, cell.weight.dtype))