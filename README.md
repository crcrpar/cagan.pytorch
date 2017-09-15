# [WIP] Conditional Analogy GAN
PyTorch implementation of CAGAN.

See, [The Conditional Analogy GAN: Swapping Fashion Articles on People Images](http://arxiv.org/abs/1709.04695)

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
