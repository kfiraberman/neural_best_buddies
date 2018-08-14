<br><br><br>

# Neural Best-Buddies in PyTorch

This is our PyTorch implementation for the Neural-Best Buddies paper. It is still under active development.

The code was written by [Kfir Aberman](https://kfiraberman.github.io/) and supported by [Mingyi Shi](https://rubbly.cn/).

**Neural Best-Buddies: [Project](http://fve.bfa.edu.cn/recap/nbbs/) |  [Paper](https://arxiv.org/pdf/1805.04140.pds) **
<img src="./images/teaser.jpg" width="800" />

If you use this code for your research, please cite:

Neural Best-Buddies: Sparse Cross-Domain Correspondence
[Kfir Aberman](https://kfiraberman.github.io/), [Jing Liao](https://liaojing.github.io/html/), [Mingyi Shi](https://rubbly.cn/), [Dani Lischinski](http://danix3d.droppages.com/), [Baoquan Chen](http://www.cs.sdu.edu.cn/~baoquan/), [Daniel Cohen-Or Lischinski](https://www.cs.tau.ac.il/~dcor/)
In SIGGRAPH 2018. [[Bibtex]](http://people.csail.mit.edu/junyanz/projects/pix2pix/pix2pix.bib)

## Prerequisites
- Linux or macOS
- Python 2 or 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Run

- Run the algorithm
```bash
#!./script.sh
python3 main.py --datarootA ./images/original_A.png --datarootB ./images/original_B.png --gpu_ids 0 --imageSize 224 --name lion_cat
```

## Citation
If you use this code for your research, please cite our papers.
```
@article{aberman2018neural,
  author = {Kfir, Aberman and Jing, Liao and Mingyi, Shi and Dani, Lischinski and Baoquan, Chen and Daniel, Cohen-Or},
  title = {Neural Best-Buddies: Sparse Cross-Domain Correspondence},
  journal = {ACM Transactions on Graphics (TOG)},
  volume = {37},
  number = {4},
  pages = {69},
  year = {2018},
  publisher = {ACM}
}

```



