# A Heuristic Exploration to Retraining-free Weight Sharing for CNN Compression


The computational workload involved in Convolutional Neural Networks (CNNs) is typically out of reach for low-power embedded devices. The scientific literature provides a large number of approximation techniques to address this problem. Among them, the Weight-Sharing (WS) technique gives promising results, but it requires to carefully determine the shared values for each layer of a given CNN. As the number of possible solutions grows exponentially with the number of layers, the WS Design Space Exploration (DSE) phase time can easily explode for state-of-the-art CNNs. This paper proposes a new heuristic approach to drastically reduce the exploration time without sacrificing the quality of the output. Results carried out on recent CNNs (Mobilenet, Resnet50V2 and Googlenet) trained with the Imagenet dataset achieved about 4 X of compression with an acceptable accuracy loss (complying with the MLPERF constraints) without any retraining step and in less than 16 hours

Code for our ASP-DAC 2022 Paper: A Heuristic Exploration to Retraining-free Weight Sharing for CNN Compression
ID: 1174
Etienne Dupuis, David Novo, Ian O'Connor, Alberto Bosio

## Update

10/19/2021: Wait for our institution approval to publish the code

