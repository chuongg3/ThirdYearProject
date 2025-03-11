# ThirdYearProject
This repository is used to contain all the files used to collect data and train model for my Third Year Project.
Project Goals:
 - Collect Alignment Score from previous implementations to build a model
 - Collect embeddings which can be used to represent the function's LLVM IR
 - Develop a model which is good a predicting alignment score
 - Integrate the model into LLVM to use predicted alignment score instead of manually calculating it

## Setting Up
Clone this repository using:
`git clone https://github.com/chuongg3/ThirdYearProject.git`

Go to the repository:
`cd ThirdYearProject`

Run the setup script:
'./setup.sh'

The set up script will:
 - Install Tensorflow's C API
 - Install LLVM which predicts alignment score
 - Download and set up f3m_exp running experiments with alignmentScore prediction
 - Clone IR2Vec

To build IR2Vec, please follow their guide here: [https://github.com/IITH-Compilers/IR2Vec](https://github.com/IITH-Compilers/IR2Vec/tree/main?tab=readme-ov-file#building-from-source)
 - You need to have a local build of LLVM
