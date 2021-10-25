# DLCV Final Project ( Medical-Imaging )

# How to run your code?
## Prediction:
1. To predict the testing images by the single best model (Public:0.78255, Private:0.78312)
    - Run bash predict.sh <Path_to_Blood_data> <Path_to_output_file>
    - ex. bash predict.sh ./Blood_data ./pred.csv (It will create a file named pred.csv)
2. To obtain the best results in the leaderboard (Public:0.79395, Private:0.79353)
    - Run bash ensemble.sh <Path_to_Blood_data> <Predict_Flag>
    - ex. bash ensemble.sh ./Blood_data 1
    - Available values for <Predict_Flag> is 1 or 0. If <Predict_Flag> = 1, then the script will do the predictions from beginning. If <Predict_Flag> = 0, then the script will use the .csv files which are prepared by us and do the ensemble method only.
    - We need to download multiple models from dropbox, the total size of these models is 1GB.
    - If <Predict_Flag> is 1,
        - at least 20GB GPU memory is needed!
        - After running this script, choose the file named "ensemble.csv" in "output_csv" directory and submit to the kaggle to obtain the best f2-score.
        - It needs about 40 minutes to do the predictions.

## Training:
Run the following command to train the model:
- bash download.sh
- cd src
- python3 main.py <Path_to_Blood_data> <Model_Saved_Path> (ex. python3 main.pt ./Blood_data ./model)

### Details
- At least 20GB GPU memory is needed for training
- After training, it will create 3 models at the <Model_Saved_Path> which are two pretrained models and one finetuned model. The suffix of these two pretrained models are "_1" and "_2"
- For example, if we run "python3 main.py ./Blood_data ./model", it will create "model_1", "model_2", "model" at current directory. The final finetuned model is "model".

    
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2020/medical-imaging-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://docs.google.com/presentation/d/1T8Wh9rM5zCiuMVCulDCZwX9JZZ9Mqgd0Yr3uqgPpe1I/edit?usp=sharing) to view the slides of Final Project - Medical Imaging. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `Blood_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1XY5twJuNLU-tJL-sr5-efTMPqbOtodS6/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `Blood_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

### Evaluation
We will use F2-score to evaluate your model. Please refer to the introduction ppt for more details.

# Submission Rules
### Deadline
2021/1/22 11:59 GMT+8

### Late Submission Policy
#### Late Submission is NOT allowed for final project!

### Academic Honesty
-   Taking any unfair advantages over other class members (or letting anyone do so) is strictly prohibited. Violating university policy would result in an **F** grade for this course (**NOT** negotiable).    
-   If you refer to some parts of the public code, you are required to specify the references in your report (e.g. URL to GitHub repositories).      
-   You are encouraged to discuss homework assignments with your fellow class members, but you must complete the assignment by yourself. TAs will compare the similarity of everyone’s submission. Any form of cheating or plagiarism will not be tolerated and will also result in an **F** grade for students with such misconduct.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

### Packages
This homework should be done using python3.6. For a list of packages you are allowed to import in this assignment, please refer to `requirments.txt` for more details.

You can run the following command to install all the packages listed in `requirements.txt`:

    pip3 install -r requirements.txt

Note that using packages with different versions will very likely lead to compatibility issues, so make sure that you install the correct version if one is specified above. E-mail or ask the TAs first if you want to import other packages.


# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in FB group
