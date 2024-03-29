<a name="readme-top"></a>


<!-- PROJECT LOGO -->
[![Pytorch][Pytorch.org]][PyTorch-url]

<br />
<div align="center">

  <h3 align="center">Quantum Machine Learning</h3>

  <p align="center">
    Quantum Vision Transformers

  </p>
</div>


<!-- ABOUT THE PROJECT -->
## About The Project

In this study we implemented the Quantum Orthogonal Transformer from [Quantum Vision Transformers, Cherrat et al.](https://arxiv.org/abs/2209.08167) using Pytorch and Qiskit.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

To execute this code you have to create the right virtual environment.
```
conda env create -f environment.yml
```
To note that there will be an error the first time is ran. You have to follow the error and modify a qiskit file (base_sampler.py) as suggested by the terminal

### Functionalities
Running main.py will execute the training. To see the possible arguments 
```
main.py -h
```

Or check directly the main.py file.


[PyTorch-url]: https://pytorch.org/
[Pytorch.org]:https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
