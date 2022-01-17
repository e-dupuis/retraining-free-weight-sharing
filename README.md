# A Heuristic Exploration to Retraining-free Weight Sharing for CNN Compression

The computational workload involved in Convolutional Neural Networks (CNNs) is typically out of reach for low-power
embedded devices. The scientific literature provides a large number of approximation techniques to address this problem.
Among them, the Weight-Sharing (WS) technique gives promising results, but it requires to carefully determine the shared
values for each layer of a given CNN. As the number of possible solutions grows exponentially with the number of layers,
the WS Design Space Exploration (DSE) phase time can easily explode for state-of-the-art CNNs. This paper proposes a new
heuristic approach to drastically reduce the exploration time without sacrificing the quality of the output. Results
carried out on recent CNNs (Mobilenet, Resnet50V2 and Googlenet) trained with the Imagenet dataset achieved about 4 X of
compression with an acceptable accuracy loss (complying with the MLPERF constraints) without any retraining step and in
less than 16 hours

Code for our ASP-DAC 2022 Paper: A Heuristic Exploration to Retraining-free Weight Sharing for CNN Compression ID: 1174
Etienne Dupuis, David Novo, Ian O'Connor, Alberto Bosio

## Citation

If you use any of the materials of this repo in your research or industrial project, please cite us with:

```bibtex
@article{Dupuis2022HeuristicExplo,
  title={A Heuristic Exploration of Retraining-free Weight-Sharing for CNN Compression},
  author={Etienne Dupuis and David Novo and Ian O'Connor and Alberto Bosio},
  journal={2022 ASPDAC '22: Proceedings of the 27th Asia and South Pacific Design Automation Conference},
  year={2022},
 }
```

## Update

* 10/19/2021: Wait for our institution approval to publish the code
* 01/18/2022: Publication of the repo under the Apache-2.0 license



## Compressing CNN



a GPU environment capable of running docker is recommended

### Docker (recommended)

Build Dockerfile & run compression

```bash
./run_docker.sh mobilenetv2_imagenet_pytorch
```

### Host

#### Installing requirements

Install requirements

```bash
./pip_install.sh
```

Run compression

```bash
NETWORK=mobilenetv2_imagenet_pytorch
python3 -u main.py \
   --network_dataset ${NETWORK} \
   --log_file "logs/${NETWORK}/latest" \
   --dataset_scale 0.1 \
   --with_local_optimization 1 \
   --mode exploration \
   --optimization two_step_optimization
```

For more details on the parameters

```bash
python3 -u main.py --help
```

## Selecting the CNN

The parameter ```--network_dataset``` takes one of the following values:

```python
lenet_mnist  # LeNet-5 on the MNIST dataset with tensorflow backend
resnet50v2_cifar10  # resnet50v2 on the CIFAR10 dataset with tensorflow backend
resnet18v2_cifar10  # resnet18v2 on the CIFAR10 dataset with tensorflow backend
mobilenetv1_imagenet  # mobilenetv1 on the IMAGENET dataset with tensorflow backend
mobilenetv2_imagenet  # mobilenetv2 on the IMAGENET dataset with tensorflow backend
mobilenetv2_imagenet_pytorch  # mobilenetv1 on the IMAGENET dataset with pytorch backend
mobilenetv3_small_imagenet  # mobilenetv3 small on the IMAGENET dataset with tensorflow backend
mobilenetv3_small_imagenet_pytorch  # mobilenetv3 small on the IMAGENET dataset with pytorch backend
mobilenetv3_large_imagenet  # mobilenetv3 large on the IMAGENET dataset with tensorflow backend
mobilenetv3_large_imagenet_pytorch  # mobilenetv3 large on the IMAGENET dataset with pytorch backend
efficientnetb0_imagenet  # efficientnetB0 on the IMAGENET dataset with tensorflow backend
efficientnetbX_imagenet_pytorch  # efficientnetBX on the IMAGENET dataset with pytorch backend (replace X by the number, ex:efficientnetB0 for B0)
resnet50v2_imagenet  # resnet50v2 on the IMAGENET dataset with tensorflow backend
resnet50_imagenet_pytorch  # resnet50 on the IMAGENET dataset with pytorch backend
inceptionv3_imagenet_pytorch  # inceptionV3 on the IMAGENET dataset with pytorch backend
inceptionv1_imagenet_pytorch  # googlenet/inceptionV3 on the IMAGENET dataset with pytorch backend
```

The most tested is ```mobilenetv2_imagenet_pytorch```

## License

This repo is under the Apache-2.0 license.

## Contact

Please raise an issue on Github