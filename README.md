# Personalized Federated Learning with Feature Alignment via Knowledge Distillation

- 我们提出了一种新颖的个性化联邦学习框架PFAKD，该框架通过知识蒸馏技术实现特征信息的有效传递，允许客户端同时学习个性化和全局特征信息，这在一定程度上提升了局部特征提取器的泛化能力。

- PFAKD通过实现更细粒度的特征对齐，有效地限制了局部特征提取器的多样性并促进了全局聚合，这使得客户端可以运行更多的本地更新，以通信高效的方式学习通用表示。

- 我们在多个不同的数据集和模型上进行了广泛实验，结果表明PFAKD均优于基线方法，这证明了 PFAKD 相对于最先进的方法的优越性和通用性。

对于代码部分，我们在个性化联邦学习中使用了一个开源算法库Pfllib，它集成了一些当前最先进的算法，我们的方法的代码是基于这个框架编写的。其他方法请参考PFLlib https://github.com/TsingZ0/PFLlib
