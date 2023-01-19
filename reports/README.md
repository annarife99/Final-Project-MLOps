---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Overall project checklist

The checklist is *exhaustic* which means that it includes everything that you could possible do on the project in
relation the curricilum in this course. Therefore, we do not expect at all that you have checked of all boxes at the
end of the project.

### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [x] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [x] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data version control setup
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [ ] Check how robust your model is towards data drifting
* [ ] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [x] Uploaded all your code to github

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

Group: Awasome 43

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

> *s212784, s212487, s213237*

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

For our project work, we use the Transformers pipeline as the basis of our project since we have been working with NLP. Thus, the transformers based-model has been used to classify the data. However, other frameworks have been used such as PyTorch library to be able to provide the necessary tools for building and training the NLP model using the "model" module. We also used the AdamW optimizer from PyTorch for optimization and PyTorch's built-in loss functions for computing the loss, together with scheduler provided by PyTorch for learning rate scheduling. Additionally, we used the hydra library to import the configurations from a yaml file, the transformers library for tokenization, the pandas and NumPy libraries for data manipulation, the wandb library for logging, the sklearn library for metrics, the tqdm library for progress bars, the click library to create a command-line interface, the logging library to log progress, the dotenv library to load environment variables, the pathlib library to work with file paths, the os library to join paths and the torch library for CUDA support. Finally, we used the Natural Language Toolkit (nltk) library which we used in preprocessing functions for text data such as removing emojis, numbers, punctuations, stop words, lower casing, lemmatizing the text and tokenization of text.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

Some packages were installed using pip, such as nltk, pytorch-lightning, and matplotlib, and others were installed using conda, such as torch and torchvision. The package managers allow for easy installation, updating and uninstalling of packages, also it helps in resolving dependencies. For this reason, we’ve choose to use conda as primarily package manager.  
To get the list of the packages installed with pip we used:
pip list --format=freeze > <pipfileName>.txt 
instead of:
pip freeze > <filename>
To remove odd path references and just keep the package and the version we used. 
To get the conda list we used: 
conda list –explicit >  <condafileName>.txt

Then we used:
cat <condafileName>.txt >> <pipfileName>.txt 
To join both requirements. 

Once we had both requirements in a list we checked for the essential ones, as pip freeze gives all the packages you had installed using pip management. 
To reply our environment you can just run the following command once you have installed python and conda. 

conda create -n myenv python=3.8.15
conda activate myenv
conda install pip 
go to the directory you stored the file <condafileName>.txt
python <condafileName>.txt 


### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

The overall structure from cookiecutter that we've used is:
.dvc : contain the configuration files for dvc, a version control system for machine learning projects.
.gitignore : git ignore file, used to ignore files and directories that should not be tracked by git
LICENSE : license file for the project
Makefile : a Makefile is a simple way to automate common tasks in a project.
README.md : contains information about the project
requirements.txt : contains the project dependencies
data.dvc : data version control file for dvc
data/processed/test.pth : a processed data file for the test set
data/processed/train.pth : a processed data file for the train set
data/raw/ : contains the raw data files
docs/ : contains documentation files
models/ : contains the models and their configurations
notebooks/ : contains the project's notebooks which we didn’t use.
outputs.dvc : outputs version control file for dvc
predict.dockerfile : a file that contains instructions to build a container image
pyproject.toml : a file that contains the project's dependencies
references/ : contains references used in the project
reports/ : contains the project's reports
src/ : contains the source code for the project which we used:
src/features: to cbuild the features.
src/models: to make the train model and the predict model, we have also the config files for hydra and some utils script for useful functions.
src/visualitzation: We printed some useful images of how our model performs. 
test_environment.py : a script to test the environment
tox.ini : configuration file for Tox, a tool for running tests in multiple Python environments
trainer.dockerfile : a file that contains instructions to build a container image

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We used Black and isort packages with a default configuration. Some of are:
*Black:*
-Use 4 spaces per indentation level.
-Limit all lines to a maximum of 79 characters.
-Use double quotes, not single quotes, for string literals.
-Use spaces around operators and after commas, but not directly inside parentheses or square brackets.
-Use empty lines to separate top-level function and class definitions.
*Isort:*
-Sorting imports alphabetically and by type (e.g. standard library imports, third-party imports, local imports)
-Grouping imports by type (e.g. standard library imports, third-party imports, local imports)
-Sorting imports within groups alphabetically.


## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We have implemented a total of 3 tests. First one, is primarily testing that training and testing data has the expected shape, and if each datapoint has the expected structure. It also checks that at least 5000 (train) and 500 (test) of each label are included in the data. Second file, tests if the output of the model matches the expected shape, when inputing sample data with the same shape as our data. Finally, last one kind of tests that the main idea behind the training file is done as expected, with sample hyperparameters for learning rate, batch size...

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our code is X%. 

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Each of the three members has implemented changes in the project while always using a secondary personal branch. Every time a significant change was implemented a pull request to the main branch was created, so each of us could pull the changes from the main branch and work from there again. Still not super familiar with how should be the best approach when using branches, at least we got aware of how important is their usage in order to prevent messing up with all the code in the main directory.

So far we have just created branches from the main branch, which gives kind of a really simple structure, but it could be useful to create sub-branches of the already existing branches when aiming to apply more complex and significant changes from the main branch, that should be divided in different subtasks.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

--- question 10 fill here ---

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

So far we have just had time to implement unittesting continuous integration everytime a push was being made to the main repository. 

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We have made use of config files, which inside the python script we call them in the following way: `@hydra.main(config_path="config", config_name="config.yaml")`. Get the hyparameters in the following way: `hparams = config.experiment`, and call each specific parameter like that: `batch_size = hparams["batch_size"]`. Therefore, we can just run an experiment specifying in the .yaml file which hyperparameters do we want, and the execute `python train_model.py`.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To handle reproducibility of experiments, all the hyperparameters have been defined in the .yaml file. Among all the hyperparameters defined, that allow the user to run the experiment with the exact same configuration, an hyperparameter called seed is defined also as 123.

This parameter is called inside the train_model.py file to be use with `torch.manual_seed(hparams["seed"])` function, which sets the seed for the pseudorandom number generator used by the torch library.

This allows to reproduce the same random numbers generated during the training and test phases of a model. This is useful when debugging, comparing different models and ensuring reproducibility of experiments.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

--- question 15 fill here ---

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

We got most of the bugs by using Pycharm debugger, VScode debugger (depending on the member of the group) and a lot of patience. We did not have time to implement profiling, as it took us longer than expected to just make the code work without crashing. But we are aware that our code has important limitations, and among them, it takes quite long to make it run. Thus, with more time we would implement profiling to detect where is the bottleneck in our project and start optimizing code from there.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

--- question 21 fill here ---

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

--- question 22 fill here ---

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

--- question 23 fill here ---

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

--- question 24 fill here ---

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 25 fill here ---

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

--- question 26 fill here ---

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

--- question 27 fill here ---
