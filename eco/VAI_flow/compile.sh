#!/bin/bash

# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Author: Mark Harvey, Xilinx Inc

if [ $1 = zcu102 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json
      TARGET=zcu102
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ZCU102.."
      echo "-----------------------------------------"
elif [ $1 = u50 ]; then
      ARCH=/opt/vitis_ai/compiler/arch/DPUCAHX8H/U50/arch.json
      TARGET=u50
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR ALVEO U50.."
      echo "-----------------------------------------"
else
      echo  "Target not found. Valid choices are: zcu102, u50 ..exiting"
      exit 1
fi


compile() {
  vai_c_xir \
  --xmodel      quant_model/CNN_int.xmodel \
  --arch        $ARCH \
  --net_name    eco \
  --output_dir  compiled_model \
  --options    '{"output_ops": "CNN__CNN_Sequential_layer1__BasicBlock_1__ReLU_relu__input_16_fix,CNN__CNN_Sequential_layer2__BasicBlock_1__ReLU_relu__input_29_fix,CNN__CNN_Sequential_layer3__BasicBlock_1__ReLU_relu__input_42_fix,CNN__CNN_Sequential_layer4__BasicBlock_1__ReLU_relu__483_fix"}' 
}


compile 2>&1 | tee compile.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"

