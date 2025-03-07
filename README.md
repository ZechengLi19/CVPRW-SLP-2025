# CVPRW-SLP-2025

This project hosts the implementation code for the CVPR workshop challenge:  [SLRTP at CVPR 2025](https://slrtpworkshop.github.io/).

## 🐱‍🏍 Installation
We suggest to create a new conda environment. 
```bash
# create environment
conda create --name CVPR_workshop python=3.9
conda activate CVPR_workshop
# install other relevant dependencies
pip install -r requirements.txt
```

## 📚 Preparation
The pre-extracted data is available [here](https://huggingface.co/ZechengLi19/CVPRW-SLP-2025). 

## 🚀 Training & Evaluation
If you have not downloaded the aforementioned pre-extracted data, you can reproduce the experimental results by following the procedures outlined below. If you have downloaded them, proceed directly to Step 4.
1. To train the **Gloss2Text** model, please follow the instructions provided in the [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign) repository.  
2. To train the **Sign2Gloss** model, please refer to the instructions in the [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) repository.  
3. To **construct gloss-pose paired data**, follow the instructions in the [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign) repository.  
4. Once completed, you can run the following script to obtain the final results:  
```bash
python main.py
```

## 📫 Contact
If you have any questions, please feel free to contact Zecheng Li (lizecheng19@gmail.com). Thank you.

## 👏 Acknowledgement
The codebase is adapted from [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) and [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign). 
