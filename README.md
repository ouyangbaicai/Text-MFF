# Text-MFF (Expert Systems With Applications, 2026)：

#### 权重和相关数据正在整理，预计一个月后将更新完毕所有内容。  
#### 欢迎参考和引用我们的工作(Welcome to refer to and cite our work) 
#### 文章发表在Expert Systems with Applications Volume 311, 15 May 2026上
#### Code for paper [“Text-MFF: Degradation multi-focus image fusion using multi expert text constraints”](https://www.sciencedirect.com/science/article/abs/pii/S0957417426002824).  
  
# Acknowledgments ※  
训练、测试框架由 [MFFT(EAAI, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624001258) 构建而来。  
网络结构由 [ArtFlow(CVPR, 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/An_ArtFlow_Unbiased_Image_Style_Transfer_via_Reversible_Neural_Flows_CVPR_2021_paper.html) 和 [Text-IF(CVPR, 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.html) 启发而来。  
感谢上述作者所作出的杰出工作。 
  
The training and testing framework is built by [MFFT(EAAI, 2024)](https://www.sciencedirect.com/science/article/abs/pii/S0952197624001258).  
The network structure is inspired by [ArtFlow(CVPR, 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/An_ArtFlow_Unbiased_Image_Style_Transfer_via_Reversible_Neural_Flows_CVPR_2021_paper.html) and [Text-IF(CVPR, 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.html).  
Thank you to all the authors mentioned above for their outstanding work.  

## Multi-focus image fusion related work

<div align="center">

| **Method** | **Code** | **Paper** | **Status** |
|:----------:|:--------:|:---------:|:----------:|
| **MSI-DTrans** | [<img src="https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github" alt="GitHub"/>](https://github.com/ouyangbaicai/MSI-DTrans) | [<img src="https://img.shields.io/badge/Paper-DISPLAYS-blue?style=for-the-badge" alt="Paper"/>](https://link.springer.com/article/10.1007/s10489-025-06383-8) | ✅ Published |
| **FusionGCN** | [<img src="https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github" alt="GitHub"/>](https://github.com/ouyangbaicai/FusionGCN) | [<img src="https://img.shields.io/badge/Paper-ESWA-blue?style=for-the-badge" alt="Paper"/>](https://www.sciencedirect.com/science/article/abs/pii/S0952197625026983) | ✅ Published |
| **Frame-MFF** | [<img src="https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github" alt="GitHub"/>](https://github.com/ouyangbaicai/Frame-MFF) | [<img src="https://img.shields.io/badge/Preprint-Private-orange?style=for-the-badge" alt="Preprint"/>](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5954614) | 🙅‍ Unrevealed |
| **Text-MFF** | [<img src="https://img.shields.io/badge/GitHub-Repo-blue?style=for-the-badge&logo=github" alt="GitHub"/>](https://github.com/ouyangbaicai/Text-MFF) | [<img src="https://img.shields.io/badge/Paper-ESWA-blue?style=for-the-badge" alt="Paper"/>](https://www.sciencedirect.com/science/article/abs/pii/S0957417426002824) | ✅ Published |

</div>
  
# Future Prospects  
-   The generation of statements is limited by a fixed vocabulary.
-   Only cosine similarity may mislead the network into producing incorrect statements.
-   Unable to effectively address strong and variable degradation interference.
  
# How to use ※
-   仅提供关键代码和权重。
-   完整代码构建可参考[FusionGCN](https://github.com/ouyangbaicai/FusionGCN)项目。
-   仅需简单替换即可完成。   
-   Only provide key codes and weights.  
-   The complete code construction can refer to the [FusionGCN](https://github.com/ouyangbaicai/FusionGCN) project.
-   Simply replace it to complete.

# submitted and accepted dates  
-   **ESWA-D-25-15335:** **STJ**(6.17)→**WE**(6.18)→**UR**(7.5)→**DIP**(9.15)→**Revise(9.16)**  
-   **R1:** **STJ**(9.18)→**WE**(9.18)→**UR**(9.24)→**DIP**(10.6)→**Revise(10.8)**  
-   **R2:** **STJ**(10.8)→**WE**(10.8)→**UR**(10.16)→**DIP**(12.24)→**Revise(12.26)**  
-   **R3:** **STJ**(1.7)→**WE**(1.7)→**UR**(1.11)→**DIP**(1.14)→**Revise(1.19)**
-   **R4:** **STJ**(1.19)→**WE**(1.19)→**UR**(1.22)→**DIP**(1.26)→**Acepted(1.26)**

# Reference information ※  
```  
@article{OUYANG2026131369,
title = {Text-MFF: Degradation multi-focus image fusion using multi expert text constraints},
journal = {Expert Systems with Applications},
volume = {311},
pages = {131369},
year = {2026},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2026.131369}
}
```
  
### Or  
  
```
Ouyang Y, Zhai H, Jiang J, et al. Text-MFF: Degradation multi-focus image fusion using multi expert text constraints[J]. Expert Systems with Applications, 2026: 131369.
```

# Contact information  
E-mail addresses: 2023210516060@stu.cqnu.edu.cn (Y. Ouyang)
