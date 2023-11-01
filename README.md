# EENN-AVCS
Code for paper [Anytime-Valid Confidence Sequences for Consistent Uncertainty Estimation in Early-Exit Neural Networks]().

## Setup 
1. Clone or download this repo. `cd` yourself to it's root directory.
2. Create and activate python [conda](https://www.anaconda.com/) enviromnent: `conda create --name eenn-avcs python=3.8`
3. Activate conda environment:  `conda activate eenn-avcs`
4. Install dependencies, using `pip install -r requirements.txt`

## Code
- For sythetic data experiment, see `regression_synthetic.ipynb`
- For ALBERT experiment, first download precomputed logits from [here](https://drive.google.com/drive/folders/1uDsYII_BBlpUrTweEbh33xaJCQJJJlHp?usp=sharing) and store them in `data/` folder. Then see `regression_albert.ipynb`
- For MSDNet experiment, first download precomputed logits from [here](https://drive.google.com/drive/folders/1uDsYII_BBlpUrTweEbh33xaJCQJJJlHp?usp=sharing) and store them in `data/` folder. Then see `classification_msdnet.ipynb`

## Acknowledgements
The [Robert Bosch GmbH](https://www.bosch.com) is acknowledged for financial support.
