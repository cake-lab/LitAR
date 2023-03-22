# LitAR: Visually Coherent Lighting for Mobile Augmented Reality

Authors:
- [Yiqin Zhao](https://yiqinzhao.me), Worcester Polytechnic Institute, USA
- [Chongyang Ma](http://www.chongyangma.com), Kuaishou Technology, China
- [Haibin Huang](https://brotherhuang.github.io), Kuaishou Technology, China
- [Tian Guo](https://tianguo.info), Worcester Polytechnic Institute, USA

This is the official code release for LitAR which was published in IMWUT (UbiComp) 2022.

**TL;DR**: LitAR can improve the rendering quality for Mobile AR applications via accurate and near real-time lighting estimation. LitAR's improved lighting estimation over existing methods is achieved through lightweight 3D reconstruction and edge-assisted computing. 😄

If you are interested in LitAR, feel free to check out our other works in Mobile AR:
- [Xihe](https://github.com/cake-lab/Xihe)
- [PointAR](https://github.com/cake-lab/PointAR)
- [Privacy-preserving reflection](https://arxiv.org/pdf/2207.03056.pdf)


## Paper

[LitAR: Visually Coherent Lighting for Mobile Augmented Reality](https://arxiv.org/pdf/2301.06184.pdf)

If you use the LitAR data or code, please cite:

```bibtex
@article{zhao2022litar,
  title={LITAR: Visually Coherent Lighting for Mobile Augmented Reality},
  author={Zhao, Yiqin and Ma, Chongyang and Huang, Haibin and Guo, Tian},
  journal={Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume={6},
  number={3},
  pages={1--29},
  year={2022},
  publisher={ACM New York, NY, USA}
}
```

## Directory Structure

- `server`: contains server-side code and relevant information for reproducing the server-side experimental results.
- `client`: contains an Unity3D-based application. This application was developed using LitAR client/server APIs, and can be used for reproducing the remaining experimental results.

We provide detailed instructions for reproducing results in `README.md` files in both the `server` and `client` directories.

## Acknowledgement

We thank the anonymous reviewers for their constructive reviews. This work was supported in part by National Science Foundation Grants 1815619, 2105564, and VMWare.
