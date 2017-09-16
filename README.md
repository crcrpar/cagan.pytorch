# [WIP] Conditional Analogy GAN
PyTorch implementation of CAGAN.

See, [The Conditional Analogy GAN: Swapping Fashion Articles on People Images](http://arxiv.org/abs/1709.04695)

## Requirements
- pytorch
- torchvision

## Usage
`python main.py --root <path/to/root> --base_root <dirname of dataset> --triplet <filename of triplet list>`  
This argument parser is a little confusiong. Hierarchy of dataset directory is assumed to be as below.  
```bash
- root
  - base_root
    - 0001.jpg
    - 0002.jpg
    - ...
  - triplet.json
```

## Dataset
Zalando has awesome great API which enables us to collect images.
Authors said that they used 15,000 tops images collected from [zalando.se](https://www.zalando.se/) which has the API.  
In this implementation, [The Zalando python client](http://www.pinchofintelligence.com/zalando-python-client/) is used to
collect images which we can install via PyPI, `pip install zalandoclient`.
