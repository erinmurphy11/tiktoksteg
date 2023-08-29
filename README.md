# Hiding TikToks in Plain Sight: Mobile Video Applications of Deep Steganography
This repository contains the code used for an undergraduate thesis I completed in the spring semester of 2023. 

It extends an exisiting convolutional video steganography architecture by incorporating depth-wise separable convolutions within the original model's architecture.

<b>Note:</b> the resulting model was trained on a dataset of video frames scraped form TikTok, that are not included in this GitHub repository.

## Abstract
Modern deep steganography techniques have proven to be highly effective at concealing information from the human eye within digital media, including images and videos. Given the increasing frequency of digital media being shared online, there are many opportunities for steganography to be implemented, whether for invisibly watermarking content shared on social media or for discreetly sending private information. Despite the growing usage and privacy concerns associated with mobile devices, current deep steganography techniques have several limitations preventing their use in mobile media platforms. One major limitation is that these models often require significant computational power and resources, making them impractical for use on mobile devices, which are commonly used to share data and access social media applications. As a result, there is a need for new steganography techniques specifically designed for mobile use. This thesis is an extension of the convolutional video steganography architecture proposed by Weng, et al in their paper introducing "High-Capacity Convolutional Video Steganography." This work successfully incorporates depth-wise separable convolutions within the original model’s architecture and trains the resulting model on video frames scraped from TikTok. The resulting model has 99.5% less parameters compared to the original, but still successfully reconstructs secret data hidden within realistic container images. This is a significant step toward developing deep steganography techniques that can effectively run on mobile devices and interact with social media platforms.

## Thesis
https://doi.org/10.33009/FSU_f77fccce-f93b-4769-abc4-a8b82e7e38f1
