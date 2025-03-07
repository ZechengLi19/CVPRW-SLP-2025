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
The pre-extracted data is available [here](https://huggingface.co/ZechengLi19/CVPRW-SLP-2025/tree/main). 

## 🚀 Training & Evaluation
If you have not downloaded the aforementioned pre-extracted data, you can reproduce the experimental results by following the procedures outlined below. If you have downloaded them, proceed directly to Step 5.
1. To train the **Text2Gloss** model, please follow the instructions provided in the [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign) repository.  
2. To **predict Text2Gloss result**, please follow the instructions in the [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign) repository. Specifically, you need to execute the command `python prediction.py --config=${config_file}` as instructed. Prior to this step, run `python replace_text.py` to replace the first 500 text samples in the test set with the challenging set texts, enabling the generation of corresponding text outputs (🚨 Do not forget to edit `config_file` modified path of data.test). The output results will be stored in `T2G/prediction/test/phoenix_results.pkl`. Please rename this file to `phoenix_text2gloss_results.pkl`.
3. To train the **Sign2Gloss** model, please refer to the instructions in the [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) repository.
4. To **construct the gloss-pose pairs** model, please refer to the instructions provided in the [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign) repository to obtain the `phoenix_iso.train` file. Then, run `python gen_segment.py` to generate the final gloss-pose pairs.
5. Once completed, you can run `python main.py` to obtain the final results: 


## 📫 Contact
If you have any questions, please feel free to contact Zecheng Li (lizecheng19@gmail.com). Thank you.

## 👏 Acknowledgement
The codebase is adapted from [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork) and [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign). 
