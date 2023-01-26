
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![GNU General Public License v2.0][license-shield]][license-url]

<div id="readme-top"></div>
<h3 align="center">DNN-HMM system for text-independent speaker identification</h3>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project
Deep learning approaches are progressively gaining popularity as alternative to HMM models for speaker identification. Promising results have been obtained with Convolutional Neural Networks (CNNs) fed by raw speech samples or raw spectral features, although this methodology does not fully take into account the temporal sequence in which speech is produced.
	
DNN-HMM (Deep Neural Network-Hidden Markov Model) is a methodology that combines the statistical modeling power of HMMs with the learning power of deep neural networks. While this technique has seen wide use in speech recognition field, few studies tried to apply it to speaker identification tasks.
	
This study proposes a novel approach to the DNN-HMM methodology for text-independent speaker identification, involving the use of both convolutional and Long-Short-Term-Memory (LSTM) networks, in order to extract both high-level features from the entire audio and temporal-wise features from each frame, which are then used to predict the emission probabilities of an HMM.
	
The experiments conducted on the TIMIT dataset showed very promising results, suggesting that the proposed non-sequential architecture may converge faster and perform better than other known methods, if properly tuned.

### Built With

* ![TensorFlow]
* ![Keras]
* ![NumPy]
* ![Pandas]
* ![Matplotlib]
* ![PyTorch]
* ![scikit-learn]
* ![SciPy]

## Getting started

### Prerequisites
Install the requirements using the pip utility (may require to run as sudo).

```
#PyPI
pip install -r requirements.txt
```

## Installation
Firstly clone the github repo
```
git clone https://github.com/MattiaLimone/dnn-hmm.git
```
Addittionally you have to install this library that probably has failed installing during prerequiste step, it's just a copy LPCTorch https://github.com/yliess86/LPCTorch with updated dependencies.

Use pip uitlity to install the dependency from our Repo (may require to run as sudo).
```
pip install https://github.com/Attornado/LPCTorch2/archive/refs/heads/master.zip
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

## License

Distributed under the GNU General Public License v2.0. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Mattia Limone [[Linkedin profile](https://www.linkedin.com/in/mattia-limone/)]

Andrea Terlizzi [[Send an email](mailto:andrea.terlizzi@mail.com)]

Carmine Iannotti [[Linkedin Profile](https://www.linkedin.com/in/carmine-iannotti-aa031b232/)]

Luca Strefezza

Project Link: [[https://github.com/your_username/repo_name](https://github.com/MattiaLimone/dnn-hmm)]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

[contributors-shield]: https://img.shields.io/github/contributors/MattiaLimone/dnn-hmm.svg?style=for-the-badge
[contributors-url]: https://github.com/MattiaLimone/dnn-hmm/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/MattiaLimone/dnn-hmm.svg?style=for-the-badge
[forks-url]: https://github.com/MattiaLimone/dnn-hmm/network/members
[stars-shield]: https://img.shields.io/github/stars/MattiaLimone/dnn-hmm.svg?style=for-the-badge
[stars-url]: https://github.com/MattiaLimone/dnn-hmm/stargazers
[issues-shield]: https://img.shields.io/github/issues/MattiaLimone/dnn-hmm.svg?style=for-the-badge
[issues-url]: https://github.com/MattiaLimone/dnn-hmm/issues
[license-shield]: https://img.shields.io/github/license/MattiaLimone/dnn-hmm.svg?style=for-the-badge
[license-url]: https://github.com/MattiaLimone/dnn-hmm/blob/main/LICENSE
[TensorFlow]: https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white
[Keras]: https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Pandas]: https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Matplotlib]: https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[PyTorch]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[scikit-learn]: https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
[SciPy]: https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white

