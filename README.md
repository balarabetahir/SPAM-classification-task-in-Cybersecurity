# **SPAM classification task**


The purpose of this lab is to build Machine & Deep Learning models at the interface of `NLP` and `Network Security` areas through the use of `SMS Spam Collection dataset` with helping frameworks & libraries.

After completing this lab, you will be able to:

1.  Quickly explore the SMS Spam Collection dataset and build the best models with the help of functional programming and layer-by-layer model description to solve a SPAM classification task.
2.  Show different calculated metrics of the built models.
3.  Change values of some hyperparameters for model training process improving to achieve better results.
4.  Visualize the data analysis results with various plot types.


## Agenda


*   Theory and Methods
*   General part
    *   Import required libraries and dataset
    *   Some additional & preparing actions & add functions
    *   Reading the Dataset
    *   Dataset manipulations & simple EDA
    *   Dataset size & feature names
    *   Dataset primary statistics
    *   Part A. Advanced Machine Learning for SPAM classification task
    *   Part B. Advanced Deep Learning for SPAM classification task



## Theory and Methods


The basics of natural language processing (NLP)

The data that we are going to use for this is a subset of an open source default of `SMS Spam Collection dataset`, which contains SMS text examples and its corresponding labels (or tags: `Spam` and `Ham`). The file contains one message per line. Each line consists of two columns: v1 contains the label (`ham` or `spam`) and v2 contains the raw text.

This corpus has been collected from free or free for research sources on the Internet:

*   A collection of 425 SMS spam messages was manually extracted from the Grumbletext website. This is a UK forum where cell phone users make public claims about SMS spam messages, most of them without reporting the very spam message received. The identification of spam messages texts in the claims is a very hard and time-consuming task, and it involved carefully scanning hundreds of web pages. The Grumbletext website is: [Web Link](http://www.grumbletext.co.uk/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01).
*   A subset of 3,375 SMS randomly chosen ham messages of the NUS SMS Corpus (NSC), which is a dataset of about 10,000 legitimate messages collected for research at the Department of Computer Science at the National University of Singapore. The messages largely originate from Singaporeans and mostly from students attending the University. These messages were collected from volunteers who were made aware that their contributions were going to be made publicly available. The NUS SMS Corpus is avalaible at: [Web Link](http://www.comp.nus.edu.sg/\~rpnlpir/downloads/corpora/smsCorpus/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01).
*   A list of 450 SMS ham messages collected from Caroline Tag's PhD Thesis is available at [Web Link](http://etheses.bham.ac.uk/253/1/Tagg09PhD.pdf?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01).
*   Finally, we have incorporated the SMS Spam Corpus v.0.1 Big. It has 1,002 SMS ham messages and 322 spam messages and it is public available at: [Web Link](http://www.esp.uem.es/jmgomez/smsspamcorpus/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01).

The original dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01). The creators would like to note that in case you find the dataset useful, please, make a reference to the previous paper and the [web page](http://www.dt.fee.unicamp.br/\~tiago/smsspamcollection/) in your papers, research, etc.

This work presents a number of statistics, studies and baseline results for a few machine learning methods.



## Import required libraries and dataset


You can find the main file of the dataset by this link: [https://www.kaggle.com/uciml/sms-spam-collection-dataset?select=spam.csv](https://www.kaggle.com/uciml/sms-spam-collection-dataset?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01&select=spam.csv)


Alternative URL for downloading of the dataset.



```python
!wget https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Advanced_ML_DL_spam_classification_L4/spam.csv
```

    --2025-04-15 01:43:26--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Advanced_ML_DL_spam_classification_L4/spam.csv
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104, 169.63.118.104
    Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 503663 (492K) [text/csv]
    Saving to: ‘spam.csv’
    
    spam.csv            100%[===================>] 491.86K  --.-KB/s    in 0.008s  
    
    2025-04-15 01:43:26 (57.3 MB/s) - ‘spam.csv’ saved [503663/503663]
    


Import the necessary libraries to use in this lab. We can add some aliases (such as pd, plt, np, tf) to make the libraries easier to use in our code and set a default figure size for further plots. Ignore the warnings.



```python
!pip install nltk
!pip install wordcloud
!pip install tensorflow==2.4
```

    Collecting nltk
      Downloading nltk-3.8.1-py3-none-any.whl (1.5 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m63.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: click in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from nltk) (8.1.3)
    Collecting joblib (from nltk)
      Downloading joblib-1.3.2-py3-none-any.whl (302 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m302.2/302.2 kB[0m [31m33.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting regex>=2021.8.3 (from nltk)
      Downloading regex-2024.4.16-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (761 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m761.6/761.6 kB[0m [31m62.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: tqdm in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from nltk) (4.60.0)
    Requirement already satisfied: importlib-metadata in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from click->nltk) (4.11.4)
    Requirement already satisfied: zipp>=0.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from importlib-metadata->click->nltk) (3.15.0)
    Requirement already satisfied: typing-extensions>=3.6.4 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from importlib-metadata->click->nltk) (4.5.0)
    Installing collected packages: regex, joblib, nltk
    Successfully installed joblib-1.3.2 nltk-3.8.1 regex-2024.4.16
    Collecting wordcloud
      Downloading wordcloud-1.9.4-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (488 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m488.9/488.9 kB[0m [31m39.6 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: numpy>=1.6.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from wordcloud) (1.21.6)
    Requirement already satisfied: pillow in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from wordcloud) (8.1.0)
    Requirement already satisfied: matplotlib in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from wordcloud) (3.5.3)
    Requirement already satisfied: cycler>=0.10 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from matplotlib->wordcloud) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from matplotlib->wordcloud) (4.38.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from matplotlib->wordcloud) (1.4.4)
    Requirement already satisfied: packaging>=20.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from matplotlib->wordcloud) (23.1)
    Requirement already satisfied: pyparsing>=2.2.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from matplotlib->wordcloud) (3.0.9)
    Requirement already satisfied: python-dateutil>=2.7 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from matplotlib->wordcloud) (2.8.2)
    Requirement already satisfied: typing-extensions in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->wordcloud) (4.5.0)
    Requirement already satisfied: six>=1.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from python-dateutil>=2.7->matplotlib->wordcloud) (1.16.0)
    Installing collected packages: wordcloud
    Successfully installed wordcloud-1.9.4
    Collecting tensorflow==2.4
      Downloading tensorflow-2.4.0-cp37-cp37m-manylinux2010_x86_64.whl (394.7 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m394.7/394.7 MB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting absl-py~=0.10 (from tensorflow==2.4)
      Downloading absl_py-0.15.0-py3-none-any.whl (132 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m132.0/132.0 kB[0m [31m26.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting astunparse~=1.6.3 (from tensorflow==2.4)
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting flatbuffers~=1.12.0 (from tensorflow==2.4)
      Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: google-pasta~=0.2 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow==2.4) (0.2.0)
    Collecting h5py~=2.10.0 (from tensorflow==2.4)
      Downloading h5py-2.10.0-cp37-cp37m-manylinux1_x86_64.whl (2.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.9/2.9 MB[0m [31m47.1 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
    [?25hRequirement already satisfied: keras-preprocessing~=1.1.2 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow==2.4) (1.1.2)
    Collecting numpy~=1.19.2 (from tensorflow==2.4)
      Downloading numpy-1.19.5-cp37-cp37m-manylinux2010_x86_64.whl (14.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.8/14.8 MB[0m [31m77.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
    [?25hCollecting opt-einsum~=3.3.0 (from tensorflow==2.4)
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m11.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: protobuf>=3.9.2 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow==2.4) (4.21.8)
    Collecting six~=1.15.0 (from tensorflow==2.4)
      Downloading six-1.15.0-py2.py3-none-any.whl (10 kB)
    Collecting termcolor~=1.1.0 (from tensorflow==2.4)
      Downloading termcolor-1.1.0.tar.gz (3.9 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting typing-extensions~=3.7.4 (from tensorflow==2.4)
      Downloading typing_extensions-3.7.4.3-py3-none-any.whl (22 kB)
    Requirement already satisfied: wheel~=0.35 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorflow==2.4) (0.40.0)
    Collecting wrapt~=1.12.1 (from tensorflow==2.4)
      Downloading wrapt-1.12.1.tar.gz (27 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting gast==0.3.3 (from tensorflow==2.4)
      Downloading gast-0.3.3-py2.py3-none-any.whl (9.7 kB)
    Collecting tensorboard~=2.4 (from tensorflow==2.4)
      Downloading tensorboard-2.11.2-py3-none-any.whl (6.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m6.0/6.0 MB[0m [31m80.9 MB/s[0m eta [36m0:00:00[0m:00:01[0m00:01[0m
    [?25hCollecting tensorflow-estimator<2.5.0,>=2.4.0rc0 (from tensorflow==2.4)
      Downloading tensorflow_estimator-2.4.0-py2.py3-none-any.whl (462 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m462.0/462.0 kB[0m [31m52.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting grpcio~=1.32.0 (from tensorflow==2.4)
      Downloading grpcio-1.32.0-cp37-cp37m-manylinux2014_x86_64.whl (3.8 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m3.8/3.8 MB[0m [31m97.7 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hCollecting google-auth<3,>=1.6.3 (from tensorboard~=2.4->tensorflow==2.4)
      Downloading google_auth-2.39.0-py2.py3-none-any.whl (212 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m212.3/212.3 kB[0m [31m28.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth-oauthlib<0.5,>=0.4.1 (from tensorboard~=2.4->tensorflow==2.4)
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: markdown>=2.6.8 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard~=2.4->tensorflow==2.4) (3.4.3)
    Collecting protobuf>=3.9.2 (from tensorflow==2.4)
      Downloading protobuf-3.20.3-cp37-cp37m-manylinux_2_5_x86_64.manylinux1_x86_64.whl (1.0 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.0/1.0 MB[0m [31m53.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests<3,>=2.21.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard~=2.4->tensorflow==2.4) (2.29.0)
    Requirement already satisfied: setuptools>=41.0.0 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard~=2.4->tensorflow==2.4) (67.7.2)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0 (from tensorboard~=2.4->tensorflow==2.4)
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m92.5 MB/s[0m eta [36m0:00:00[0m:00:01[0m
    [?25hCollecting tensorboard-plugin-wit>=1.6.0 (from tensorboard~=2.4->tensorflow==2.4)
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m58.0 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: werkzeug>=1.0.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from tensorboard~=2.4->tensorflow==2.4) (2.2.3)
    Collecting cachetools<6.0,>=2.0.0 (from google-auth<3,>=1.6.3->tensorboard~=2.4->tensorflow==2.4)
      Downloading cachetools-5.5.2-py3-none-any.whl (10 kB)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from google-auth<3,>=1.6.3->tensorboard~=2.4->tensorflow==2.4) (0.3.0)
    Collecting rsa<5,>=3.1.4 (from google-auth<3,>=1.6.3->tensorboard~=2.4->tensorflow==2.4)
      Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Collecting requests-oauthlib>=0.7.0 (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.4->tensorflow==2.4)
      Downloading requests_oauthlib-2.0.0-py2.py3-none-any.whl (24 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from markdown>=2.6.8->tensorboard~=2.4->tensorflow==2.4) (4.11.4)
    Requirement already satisfied: charset-normalizer<4,>=2 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow==2.4) (3.1.0)
    Requirement already satisfied: idna<4,>=2.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow==2.4) (3.4)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow==2.4) (1.26.15)
    Requirement already satisfied: certifi>=2017.4.17 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from requests<3,>=2.21.0->tensorboard~=2.4->tensorflow==2.4) (2023.5.7)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from werkzeug>=1.0.1->tensorboard~=2.4->tensorflow==2.4) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard~=2.4->tensorflow==2.4) (3.15.0)
    Requirement already satisfied: pyasn1<0.6.0,>=0.4.6 in /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard~=2.4->tensorflow==2.4) (0.5.0)
    Collecting oauthlib>=3.0.0 (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.4->tensorflow==2.4)
      Downloading oauthlib-3.2.2-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.7/151.7 kB[0m [31m24.7 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: termcolor, wrapt
      Building wheel for termcolor (setup.py) ... [?25ldone
    [?25h  Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4832 sha256=8ec44d9e0cb7a2a7672bf744d026cb0ed810151cf89acc5cafbbbb7d85a7d5b1
      Stored in directory: /home/jupyterlab/.cache/pip/wheels/3f/e3/ec/8a8336ff196023622fbcb36de0c5a5c218cbb24111d1d4c7f2
      Building wheel for wrapt (setup.py) ... [?25ldone
    [?25h  Created wheel for wrapt: filename=wrapt-1.12.1-cp37-cp37m-linux_x86_64.whl size=36029 sha256=e3e1e59aa1e3f170d704708aec5b703dd3df3f962692a9f8cdd28158ab0ea5e5
      Stored in directory: /home/jupyterlab/.cache/pip/wheels/62/76/4c/aa25851149f3f6d9785f6c869387ad82b3fd37582fa8147ac6
    Successfully built termcolor wrapt
    Installing collected packages: wrapt, typing-extensions, termcolor, tensorflow-estimator, tensorboard-plugin-wit, flatbuffers, tensorboard-data-server, six, rsa, protobuf, oauthlib, numpy, gast, cachetools, requests-oauthlib, opt-einsum, h5py, grpcio, google-auth, astunparse, absl-py, google-auth-oauthlib, tensorboard, tensorflow
      Attempting uninstall: wrapt
        Found existing installation: wrapt 1.14.1
        Uninstalling wrapt-1.14.1:
          Successfully uninstalled wrapt-1.14.1
      Attempting uninstall: typing-extensions
        Found existing installation: typing_extensions 4.5.0
        Uninstalling typing_extensions-4.5.0:
          Successfully uninstalled typing_extensions-4.5.0
      Attempting uninstall: termcolor
        Found existing installation: termcolor 2.3.0
        Uninstalling termcolor-2.3.0:
          Successfully uninstalled termcolor-2.3.0
      Attempting uninstall: tensorflow-estimator
        Found existing installation: tensorflow-estimator 1.14.0
        Uninstalling tensorflow-estimator-1.14.0:
          Successfully uninstalled tensorflow-estimator-1.14.0
      Attempting uninstall: six
        Found existing installation: six 1.16.0
        Uninstalling six-1.16.0:
          Successfully uninstalled six-1.16.0
      Attempting uninstall: protobuf
        Found existing installation: protobuf 4.21.8
        Uninstalling protobuf-4.21.8:
          Successfully uninstalled protobuf-4.21.8
      Attempting uninstall: numpy
        Found existing installation: numpy 1.21.6
        Uninstalling numpy-1.21.6:
          Successfully uninstalled numpy-1.21.6
      Attempting uninstall: gast
        Found existing installation: gast 0.5.3
        Uninstalling gast-0.5.3:
          Successfully uninstalled gast-0.5.3
      Attempting uninstall: h5py
        Found existing installation: h5py 2.8.0
        Uninstalling h5py-2.8.0:
          Successfully uninstalled h5py-2.8.0
      Attempting uninstall: grpcio
        Found existing installation: grpcio 1.48.1
        Uninstalling grpcio-1.48.1:
          Successfully uninstalled grpcio-1.48.1
      Attempting uninstall: absl-py
        Found existing installation: absl-py 1.4.0
        Uninstalling absl-py-1.4.0:
          Successfully uninstalled absl-py-1.4.0
      Attempting uninstall: tensorboard
        Found existing installation: tensorboard 1.14.0
        Uninstalling tensorboard-1.14.0:
          Successfully uninstalled tensorboard-1.14.0
      Attempting uninstall: tensorflow
        Found existing installation: tensorflow 1.14.0
        Uninstalling tensorflow-1.14.0:
          Successfully uninstalled tensorflow-1.14.0
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    rich 13.3.5 requires typing-extensions<5.0,>=4.0.0; python_version < "3.9", but you have typing-extensions 3.7.4.3 which is incompatible.[0m[31m
    [0mSuccessfully installed absl-py-0.15.0 astunparse-1.6.3 cachetools-5.5.2 flatbuffers-1.12 gast-0.3.3 google-auth-2.39.0 google-auth-oauthlib-0.4.6 grpcio-1.32.0 h5py-2.10.0 numpy-1.19.5 oauthlib-3.2.2 opt-einsum-3.3.0 protobuf-3.20.3 requests-oauthlib-2.0.0 rsa-4.9 six-1.15.0 tensorboard-2.11.2 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.4.0 tensorflow-estimator-2.4.0 termcolor-1.1.0 typing-extensions-3.7.4.3 wrapt-1.12.1



```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import nltk, re, collections, pickle, os # nltk - Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# %matplotlib inline
plt.rcParams['figure.figsize'] = (15, 5)
plt.style.use('ggplot')
seed = 42

import warnings
warnings.filterwarnings(action = "ignore")
warnings.simplefilter(action = 'ignore', category = Warning)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('punkt')

```

    /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/sklearn/utils/validation.py:37: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      LARGE_SPARSE_SUPPORTED = LooseVersion(scipy_version) >= '0.14.0'
    2025-04-15 01:45:45.161832: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2025-04-15 01:45:45.161893: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
    [nltk_data] Downloading package stopwords to
    [nltk_data]     /home/jupyterlab/nltk_data...
    [nltk_data]   Unzipping corpora/stopwords.zip.
    [nltk_data] Downloading package wordnet to
    [nltk_data]     /home/jupyterlab/nltk_data...
    [nltk_data] Downloading package punkt to /home/jupyterlab/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.





    True



## Some additional & preparing actions & add functions


Specify the value of the `precision` parameter equal to 3 to display three decimal signs (instead of 6 as default).



```python
pd.set_option("precision", 3)
pd.options.display.float_format = '{:.3f}'.format
```

Add some functions that you will need futher.



```python
def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'accuracy' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'accuracy' in s and 'val' in s]
    
    plt.figure(figsize = (12, 5), dpi = 100)
    COLOR = 'gray'
    
    plt.rc('legend', fontsize = 14)   # legend fontsize
    plt.rc('figure', titlesize = 12)  # fontsize of the figure title
        
    if len(loss_list) == 0:
        print('Loss is missing in history')
        return 
    
    ## As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)
    
    ## Loss
    plt.subplot(1, 2, 1)
    plt.subplots_adjust(wspace = 2, hspace = 2)
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b-o',
                 label = 'Train (' + str(str(format(history.history[l][-1],'.4f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label = 'Valid (' + str(str(format(history.history[l][-1],'.4f'))+')'))
    
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend(facecolor = 'gray', loc = 'best')
    plt.grid(True)
    plt.tight_layout()
    
    ## Accuracy
    plt.subplot(1, 2, 2)
    plt.subplots_adjust(wspace = 2, hspace = 2)
    plt.rcParams['text.color'] = 'black'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b-o',
                 label = 'Train (' + str(format(history.history[l][-1],'.4f'))+')')
    for l in val_acc_list:    
        plt.plot(epochs, history.history[l], 'g',
                 label = 'Valid (' + str(format(history.history[l][-1],'.4f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(facecolor = 'gray', loc = 'best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_conf_matr(conf_matr, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.winter):
  """
  Citation
  ---------
  http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

  """  
  import itertools

  accuracy = np.trace(conf_matr) / np.sum(conf_matr).astype('float')
  sns.set(font_scale = 1.4)

  plt.figure(figsize = (12, 8))
  plt.imshow(conf_matr, interpolation = 'nearest', cmap = cmap)
  title = '\n' + title + '\n'
  plt.title(title)
  plt.colorbar()

  if classes is not None:
      tick_marks = np.arange(len(classes))
      plt.xticks(tick_marks, classes, rotation = 45)
      plt.yticks(tick_marks, classes)

  if normalize:
      conf_matr = conf_matr.astype('float') / conf_matr.sum(axis = 1)[:, np.newaxis]


  thresh = conf_matr.max() / 1.5 if normalize else conf_matr.max() / 2
  for i, j in itertools.product(range(conf_matr.shape[0]), range(conf_matr.shape[1])):
      if normalize:
          plt.text(j, i, "{:0.2f}%".format(conf_matr[i, j] * 100),
                    horizontalalignment = "center",
                    fontweight = 'bold',
                    color = "white" if conf_matr[i, j] > thresh else "black")
      else:
          plt.text(j, i, "{:,}".format(conf_matr[i, j]),
                    horizontalalignment = "center",
                    fontweight = 'bold',
                    color = "white" if conf_matr[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label\n\nAccuracy = {:0.2f}%; Error = {:0.2f}%'.format(accuracy * 100, (1 - accuracy) * 100))
  plt.show()


def plot_words(set, number):
  words_counter = collections.Counter([word for sentence in set for word in sentence.split()]) # finding words along with count
  most_counted = words_counter.most_common(number)
  most_count = pd.DataFrame(most_counted, columns = ["Words", "Amount"]).sort_values(by = "Amount") # sorted data frame
  most_count.plot.barh(x = "Words", 
                       y = "Amount",
                       color = "blue",
                       figsize = (10, 15))
  for i, v in enumerate(most_count["Amount"]):
    plt.text(v, i,
             " " + str(v),
             color = 'black',
             va = 'center',
             fontweight = 'bold')

def word_cloud(tag):
  df_words_nl = ' '.join(list(df_spam[df_spam['feature'] == tag]['message']))
  df_wc_nl = WordCloud(width = 600, height = 512).generate(df_words_nl)
  plt.figure(figsize = (13, 9), facecolor = 'k')
  plt.imshow(df_wc_nl)
  plt.axis('off')
  plt.tight_layout(pad = 1)
  plt.show()
```

## Reading the Dataset


The files contain one message per line. Each line consists of two columns: v1 contains the label (`ham` or `spam`) and v2 contains the raw text. SMS spam (sometimes called cell phone spam) is any junk message delivered to a mobile phone as a text messaging through the Short Message Service (SMS). The practice is fairly rare in `North America` but has been common in `Japan` for years.

***

In this section you will read our dataset.



```python
df_spam = pd.read_csv('spam.csv', encoding = 'latin-1')
```

## Dataset manipulations & simple EDA


To make the columns(v1, v2) easy to read, we can rename them respectively.



```python
df_spam = df_spam.filter(['v1', 'v2'], axis = 1)
df_spam.columns = ['feature', 'message']
df_spam.drop_duplicates(inplace = True, ignore_index = True)
print('Number of null values:\n')
df_spam.isnull().sum()
```

    Number of null values:
    





    feature    0
    message    0
    dtype: int64



Total ham(0) and spam(1) messages.



```python

df_spam['feature'].value_counts()

```




    ham     4516
    spam     653
    Name: feature, dtype: int64



## Dataset size & feature names



```python
df_spam.shape, df_spam.columns
```




    ((5169, 2), Index(['feature', 'message'], dtype='object'))



The dataset contains a lot of objects (rows), including 1 target feature (`feature`) and an additional column (`message`).


Input features (column names):

1.  `feature` - tags in this data collection
2.  `message` - raw test message example


Let's describe the data in a transposed way using both `describe` & `T` methods. The number of statistical output parameters from the data set is determined by the `describe` method.

Replace `##YOUR CODE GOES HERE##` with your Python code.



```python
df_spam.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>feature</th>
      <td>5169</td>
      <td>2</td>
      <td>ham</td>
      <td>4516</td>
    </tr>
    <tr>
      <th>message</th>
      <td>5169</td>
      <td>5169</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Dataset primary statistics


Let’s plot the number of value of both `spam` and `ham` messages.



```python
plt.figure(figsize = (10, 6))
counter = df_spam.shape[0]
ax1 = sns.countplot(df_spam['feature'])
ax2 = ax1.twinx()                      # Make double axis
ax2.yaxis.tick_left()                 # Switch so the counter's axis is on the right, frequency axis is on the left
ax1.yaxis.tick_right()
ax1.yaxis.set_label_position('right')  # Also switch the labels over
ax2.yaxis.set_label_position('left')
ax2.set_ylabel('frequency, %')


for p in ax1.patches:
  x = p.get_bbox().get_points()[:, 0]
  y = p.get_bbox().get_points()[1, 1]
  ax1.annotate('{:.2f}%'.format(100. * y / counter),
              (x.mean(), y),
              ha = 'center',
              va = 'bottom')

# Use a LinearLocator to ensure the correct number of ticks
ax1.yaxis.set_major_locator(ticker.LinearLocator(11))

# Fix the frequency range to 0-100
ax2.set_ylim(0, 100)
ax1.set_ylim(0, counter)

# And use a MultipleLocator to ensure a tick spacing of 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

# Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
ax2.grid(None)
```


    
![png](output_34_0.png)
    


The number of `ham` messages is almost for times bigger than that of `spam` messages in the data.

Let’s plot the `number` (you can choose a number of words yourself in the range of `[5 .. 50]` (it should be divisible by `5`)) of different most often used words (while we can name these objects as words) present in our dataset.



```python
plot_words(df_spam['message'], number = 30)
```


    
![png](output_36_0.png)
    


As you can see, the most often used words are `stopwords`. So we need to perform some preprocessing techniques on the dataset (see below [lemmatization](https://en.wikipedia.org/wiki/Lemmatisation?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01) substep in `Stage I`).

Let's build the `WordCloud` image for the `spam` and the existed words (label `ham`) separately.



```python
word_cloud('spam')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_337/452434762.py in <module>
    ----> 1 word_cloud('spam')
    

    /tmp/ipykernel_337/3074800124.py in word_cloud(tag)
        126 def word_cloud(tag):
        127   df_words_nl = ' '.join(list(df_spam[df_spam['feature'] == tag]['message']))
    --> 128   df_wc_nl = WordCloud(width = 600, height = 512).generate(df_words_nl)
        129   plt.figure(figsize = (13, 9), facecolor = 'k')
        130   plt.imshow(df_wc_nl)


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate(self, text)
        640         self
        641         """
    --> 642         return self.generate_from_text(text)
        643 
        644     def _check_generated(self):


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate_from_text(self, text)
        622         """
        623         words = self.process_text(text)
    --> 624         self.generate_from_frequencies(words)
        625         return self
        626 


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate_from_frequencies(self, frequencies, max_font_size)
        452             else:
        453                 self.generate_from_frequencies(dict(frequencies[:2]),
    --> 454                                                max_font_size=self.height)
        455                 # find font sizes
        456                 sizes = [x[1] for x in self.layout_]


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate_from_frequencies(self, frequencies, max_font_size)
        509                     font, orientation=orientation)
        510                 # get size of resulting text
    --> 511                 box_size = draw.textbbox((0, 0), word, font=transposed_font, anchor="lt")
        512                 # find possible places using integral image:
        513                 result = occupancy.sample_position(box_size[3] + self.margin,


    ~/conda/envs/python/lib/python3.7/site-packages/PIL/ImageDraw.py in textbbox(self, xy, text, font, anchor, spacing, align, direction, features, language, stroke_width, embedded_color)
        565             font = self.getfont()
        566         mode = "RGBA" if embedded_color else self.fontmode
    --> 567         bbox = font.getbbox(
        568             text, mode, direction, features, language, stroke_width, anchor
        569         )


    AttributeError: 'TransposedFont' object has no attribute 'getbbox'



```python
word_cloud('ham')
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_337/3063128479.py in <module>
    ----> 1 word_cloud('ham')
    

    /tmp/ipykernel_337/3074800124.py in word_cloud(tag)
        126 def word_cloud(tag):
        127   df_words_nl = ' '.join(list(df_spam[df_spam['feature'] == tag]['message']))
    --> 128   df_wc_nl = WordCloud(width = 600, height = 512).generate(df_words_nl)
        129   plt.figure(figsize = (13, 9), facecolor = 'k')
        130   plt.imshow(df_wc_nl)


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate(self, text)
        640         self
        641         """
    --> 642         return self.generate_from_text(text)
        643 
        644     def _check_generated(self):


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate_from_text(self, text)
        622         """
        623         words = self.process_text(text)
    --> 624         self.generate_from_frequencies(words)
        625         return self
        626 


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate_from_frequencies(self, frequencies, max_font_size)
        452             else:
        453                 self.generate_from_frequencies(dict(frequencies[:2]),
    --> 454                                                max_font_size=self.height)
        455                 # find font sizes
        456                 sizes = [x[1] for x in self.layout_]


    ~/conda/envs/python/lib/python3.7/site-packages/wordcloud/wordcloud.py in generate_from_frequencies(self, frequencies, max_font_size)
        509                     font, orientation=orientation)
        510                 # get size of resulting text
    --> 511                 box_size = draw.textbbox((0, 0), word, font=transposed_font, anchor="lt")
        512                 # find possible places using integral image:
        513                 result = occupancy.sample_position(box_size[3] + self.margin,


    ~/conda/envs/python/lib/python3.7/site-packages/PIL/ImageDraw.py in textbbox(self, xy, text, font, anchor, spacing, align, direction, features, language, stroke_width, embedded_color)
        565             font = self.getfont()
        566         mode = "RGBA" if embedded_color else self.fontmode
    --> 567         bbox = font.getbbox(
        568             text, mode, direction, features, language, stroke_width, anchor
        569         )


    AttributeError: 'TransposedFont' object has no attribute 'getbbox'


## Part A. Advanced Machine Learning for SPAM classification task


### I stage. Preliminary actions. Preparing of needed sets.


We need to define some input parameters for our next research, such as the `size of vocabulary`, sizes of `test` & `validation` sets, `dropping level`, etc. You can change some of those numerical parameters yourself (only where you can see the additional comments).



```python
size_vocabulary = 1000    # You can choose the size of vocabulary yourself in the range [500 .. 1500] (it should be divisible by 500)
embedding_dimension = 64  # You can choose the size of dimention yourself in the range [32 .. 256] (it should be divisible by 32)
trunc_type = 'post'
padding_type = 'post'
threshold = 0.5           # You can choose the size of threshold yourself in the range [0 .. 1]
oov_token = "<OOV>"
test_size, valid_size = 0.05, 0.2
num_epochs = 20           # You can choose the number of epochs yourself in the range [20 .. 50] (it should be divisible by 5)
drop_level = 0.3          # You can choose the size of drop level yourself in the range [0 .. 1]
```

Next actions allow you to make data cleaning step by step, which consists of the following replace rules:

1.  email addresses with 'emailaddr';
2.  URLs with 'httpaddr';
3.  money symbols with 'moneysymb';
4.  phone numbers with 'phonenumbr';
5.  numbers with 'numbr';
6.  remove all punctuations;
7.  word to lower case.

Moreover, we make a `lemmatization` which is a method of morphological analysis. It comes down to reducing a word form to its initial dictionary form (lemma). As a result of  word forms lemmatization, flexive endings are discarded and the main or dictionary form of the word is returned.



```python
print("\t\tStage I. Preliminary actions. Preparing of needed sets\n")
full_df_l = []
lemmatizer = WordNetLemmatizer()
for i in range(df_spam.shape[0]):
    mess_1 = df_spam.iloc[i, 1]
    mess_1 = re.sub('\b[\w\-.]+?@\w+?\.\w{2,4}\b', 'emailaddr', mess_1)
    mess_1 = re.sub('(http[s]?\S+)|(\w+\.[A-Za-z]{2,4}\S*)', 'httpaddr', mess_1) 
    mess_1 = re.sub('£|\$', 'moneysymb', mess_1) 
    mess_1 = re.sub('\b(\+\d{1,2}\s)?\d?[\-(.]?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b', 'phonenumbr', mess_1) 
    mess_1 = re.sub('\d+(\.\d+)?', 'numbr', mess_1) 
    mess_1 = re.sub('[^\w\d\s]', ' ', mess_1) 
    mess_1 = re.sub('[^A-Za-z]', ' ', mess_1).lower() 
    token_messages = word_tokenize(mess_1)
    mess = []
    for word in token_messages:
        if word not in set(stopwords.words('english')):
            mess.append(lemmatizer.lemmatize(word))
    txt_mess = " ".join(mess)
    full_df_l.append(txt_mess)
```

    		Stage I. Preliminary actions. Preparing of needed sets
    


Now, let’s plot the count words (`number` - you can choose a number of words yourself in the range `[5 .. 50]` (it should be divisible by `5`)) once again to see the most frequent words (without any stopwords, thus after all cleaning stages).



```python
plot_words(full_df_l, number = 35)
```


    
![png](output_47_0.png)
    


We can see that most common words are different from the stopwords. In addition, you can compare this picture with the result in the chapter `"Dataset primary statistics"`.

Then the primary `df_spam` set will split into sentences (messages) and labels separately. Then we will split the full primary `df_spam` set with the following proportions: a training set (`75%`) and a test set (`25%`). Thus we will obtain 4 sets: two for sentences and two for labels with the same proportions.

Also, we will do the `vectorization` with the help of `CountVectorizer` method. This is an easy way to make a collection of text documents and create a dictionary of famous words. This method converts the input text to the matrix, the values of which are the numbers of this key entry (words) in the text. Unfortunately, `FeatureHasher` has more configurable parameters (for example, you can set the tokenizer), but it works slowlier.



```python
add_df = CountVectorizer(max_features = size_vocabulary)
X = add_df.fit_transform(full_df_l).toarray()
y = df_spam.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = (test_size + valid_size), random_state = seed)
print('Number of rows in test set: ' + str(X_test.shape))
print('Number of rows in training set: ' + str(X_train.shape))
```

    Number of rows in test set: (1293, 1000)
    Number of rows in training set: (3876, 1000)


### II stage. Naive Bayes Classifier.


Let's find a set of predictions based on our models: `Guassian Naive Bayes` and `Multinomial Naive Bayes`. In addition, we will build a `classification report` and draw the `confusion matrix`.



```python
print("\t\tStage IIa. Guassian Naive Bayes\n")
class_NBC = GaussianNB().fit(X_train, y_train) # Guassian Naive Bayes
y_pred_NBC = class_NBC.predict(X_test)
print('The first two predicted labels:', y_pred_NBC[0],y_pred_NBC[1], '\n')
conf_m_NBC = confusion_matrix(y_test, y_pred_NBC)
class_rep_NBC = classification_report(y_test, y_pred_NBC)
print('\t\t\tClassification report:\n\n', class_rep_NBC, '\n')
plot_conf_matr(conf_m_NBC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Guassian Naive Bayes')
```

    		Stage IIa. Guassian Naive Bayes
    
    The first two predicted labels: spam ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.99      0.77      0.86      1107
            spam       0.40      0.94      0.56       186
    
       micro avg       0.79      0.79      0.79      1293
       macro avg       0.69      0.85      0.71      1293
    weighted avg       0.90      0.79      0.82      1293
     
    



    
![png](output_52_1.png)
    



```python
print("\t\tStage IIb. Multinomial Naive Bayes\n")
class_MNB = MultinomialNB().fit(X_train, y_train) # Multinomial Naive Bayes
y_pred_MNB = class_MNB.predict(X_test)
print('The first two predicted labels:', y_pred_MNB[0],y_pred_MNB[1], '\n')
conf_m_MNB = confusion_matrix(y_test, y_pred_MNB)
class_rep_MNB = classification_report(y_test, y_pred_MNB)
print('\t\t\tClassification report:\n\n', class_rep_MNB, '\n')
plot_conf_matr(conf_m_MNB, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Multinomial Naive Bayes')
```

    		Stage IIb. Multinomial Naive Bayes
    
    The first two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.99      0.98      0.98      1107
            spam       0.90      0.92      0.91       186
    
       micro avg       0.97      0.97      0.97      1293
       macro avg       0.94      0.95      0.95      1293
    weighted avg       0.97      0.97      0.97      1293
     
    



    
![png](output_53_1.png)
    


### III stage. Decision Tree Classifier.


Let's find a set of predictions based on our `Decision Tree Classifier` model. In addition, we will build a `classification report` and draw the `confusion matrix`.



```python
print("\t\tStage III. Decision Tree Classifier\n")
class_DTC = DecisionTreeClassifier(random_state = seed).fit(X_train, y_train)
y_pred_DTC = class_DTC.predict(X_test)
print('The first two predicted labels:', y_pred_DTC[0], y_pred_DTC[1], '\n')
conf_m_DTC = confusion_matrix(y_test, y_pred_DTC)
class_rep_DTC = classification_report(y_test, y_pred_DTC)
print('\t\t\tClassification report:\n\n', class_rep_DTC, '\n')
plot_conf_matr(conf_m_DTC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Decision Tree')
```

    		Stage III. Decision Tree Classifier
    
    The first two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.97      0.98      0.98      1107
            spam       0.87      0.84      0.86       186
    
       micro avg       0.96      0.96      0.96      1293
       macro avg       0.92      0.91      0.92      1293
    weighted avg       0.96      0.96      0.96      1293
     
    



    
![png](output_56_1.png)
    


### IV stage. Logistic Regression.


Let's find a set of predictions based on our `Logistic Regression` model. In addition, we will build a `classification report` and draw the `confusion matrix`.



```python
print("\t\tStage IV. Logistic Regression\n")
class_LR = LogisticRegression(random_state = seed, solver = 'liblinear').fit(X_train, y_train)
y_pred_LR = class_LR.predict(X_test)
print('The first two predicted labels:', y_pred_LR[0], y_pred_LR[1], '\n')
conf_m_LR = confusion_matrix(y_test, y_pred_LR)
class_rep_LR = classification_report(y_test, y_pred_LR)
print('\t\t\tClassification report:\n\n', class_rep_LR, '\n')
plot_conf_matr(conf_m_LR, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Logistic Regression')
```

    		Stage IV. Logistic Regression
    
    The first two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.98      1.00      0.99      1107
            spam       0.97      0.86      0.91       186
    
       micro avg       0.98      0.98      0.98      1293
       macro avg       0.97      0.93      0.95      1293
    weighted avg       0.98      0.98      0.98      1293
     
    



    
![png](output_59_1.png)
    


### V stage. KNeighbors Classifier.


Let's find a set of predictions based on our `KNeighbors Classifier` model. In addition, we will build a `classification report` and draw the `confusion matrix`.



```python
print("\t\tStage V. KNeighbors Classifier\n")
class_KNC = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
y_pred_KNC = class_KNC.predict(X_test)
print('The firs two predicted labels:', y_pred_KNC[0], y_pred_KNC[1], '\n')
conf_m_KNC = confusion_matrix(y_test, y_pred_KNC)
class_rep_KNC = classification_report(y_test, y_pred_KNC)
print('\t\t\tClassification report:\n\n', class_rep_KNC, '\n')
plot_conf_matr(conf_m_KNC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for KNeighbors Classifier')
```

    		Stage V. KNeighbors Classifier
    
    The firs two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.95      0.99      0.97      1107
            spam       0.93      0.70      0.80       186
    
       micro avg       0.95      0.95      0.95      1293
       macro avg       0.94      0.85      0.89      1293
    weighted avg       0.95      0.95      0.95      1293
     
    



    
![png](output_62_1.png)
    


### VI stage. Support Vector Classification.


Let's find a set of predictions based on our `Support Vector Classification` model. In addition, we will build a `classification report` and draw the `confusion matrix`.



```python
print("\t\tStage VI. Support Vector Classification\n")
class_SVC = SVC(probability = True, random_state = seed).fit(X_train, y_train)
y_pred_SVC = class_SVC.predict(X_test)
print('The first two predicted labels:', y_pred_SVC[0], y_pred_SVC[1], '\n')
conf_m_SVC = confusion_matrix(y_test, y_pred_SVC)
class_rep_SVC = classification_report(y_test, y_pred_SVC)
print('\t\t\tClassification report:\n\n', class_rep_SVC, '\n')
plot_conf_matr(conf_m_SVC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for SVC Classifier')
```

    		Stage VI. Support Vector Classification
    
    The first two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.94      0.99      0.96      1107
            spam       0.95      0.59      0.73       186
    
       micro avg       0.94      0.94      0.94      1293
       macro avg       0.94      0.79      0.85      1293
    weighted avg       0.94      0.94      0.93      1293
     
    



    
![png](output_65_1.png)
    


### VII stage. Gradient Boosting Classifier.


Let's find a set of predictions based on our `Gradient Boosting Classifier` model. In addition, we will build a `classification report` and draw the `confusion matrix`.



```python
print("\t\tStage VII. Gradient Boosting Classifier\n")
class_GBC = GradientBoostingClassifier(random_state = seed).fit(X_train, y_train)
y_pred_GBC = class_GBC.predict(X_test)
print('The first two predicted labels:', y_pred_GBC[0], y_pred_GBC[1], '\n')
conf_m_GBC = confusion_matrix(y_test, y_pred_GBC)
class_rep_GBC = classification_report(y_test, y_pred_GBC)
print('\t\t\tClassification report:\n\n', class_rep_GBC, '\n')
plot_conf_matr(conf_m_GBC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Gradient Boosting Classifier')
```

    		Stage VII. Gradient Boosting Classifier
    
    The first two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.97      1.00      0.99      1107
            spam       0.99      0.84      0.91       186
    
       micro avg       0.98      0.98      0.98      1293
       macro avg       0.98      0.92      0.95      1293
    weighted avg       0.98      0.98      0.97      1293
     
    



    
![png](output_68_1.png)
    


### VIII stage. Bagging Classifier.


Let's find a set of predictions based on our `Bagging Classifier` model. In addition, we will build a `classification report` and draw the `confusion matrix`.





```python

print("\t\tStage VIII. Bagging Classifier + something else\n")
class_BC = BaggingClassifier(class_SVC).fit(X_train, y_train)
y_pred_BC = class_BC.predict(X_test)
print('The first two predicted labels:', y_pred_BC[0], y_pred_BC[1], '\n')
conf_m_BC = confusion_matrix(y_test, y_pred_BC)
class_rep_BC = classification_report(y_test, y_pred_BC)
print('\t\t\tClassification report:\n\n', class_rep_BC, '\n')
plot_conf_matr(conf_m_BC, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for Bagging Classifier')
```

    		Stage VIII. Bagging Classifier + something else
    
    The first two predicted labels: ham ham 
    
    			Classification report:
    
                   precision    recall  f1-score   support
    
             ham       0.96      0.99      0.98      1107
            spam       0.94      0.78      0.85       186
    
       micro avg       0.96      0.96      0.96      1293
       macro avg       0.95      0.89      0.91      1293
    weighted avg       0.96      0.96      0.96      1293
     
    



    
![png](output_71_1.png)
    


Moreover, as you can see, the `Bugging Classifier` can work with some different classifiers as with basic ones, such as `SVC`, `KNC`, `DTC`, etc. In this case, the main purpose of its usage is to increase the accuracy obtained earlier from the basic classifier. You can check this fact comparing, for instance, the obtained earlier `SVC classifier` accuracy and the accuracy after using `Bugging Classifier` with `SVC`.


## Part B. Advanced Deep Learning for SPAM classification task


### I stage. Preliminary actions. Preparing of needed sets.


We need to prepare our sets to the new DL model for SPAM classification task, such as training, validation & test sets based on the primary `df_spam` set. We need a training set for training a pre-built model, a validation set is used for finding better hyperparameters, a test set will be used for checking our trained model on data which the model didn't see.

Firstly, the primary `df_spam` set will split into sentences (messages) and labels separately. Then we will split the full primary `df_spam` set with the following proportions: a training set (`75%`), a validation set (`20%`) and a test set (`5%`). Thus we will obtain 6 sets: three for sentences and three for labels with the same proportions.

Nevertheless, you can change those proportions in percentages as you see it. However, you should remember that these changes can influence your model accuracy, and as it often happens, they decrease it.



```python
print("Stage I. Preliminary actions. Preparing of needed sets\n")

sentences_new_set = []
labels_new_set = []
for i in range(0, df_spam.shape[0], 1):
    sentences_new_set.append(df_spam['message'][i])
    labels_new_set.append(df_spam['feature'][i])
```

    Stage I. Preliminary actions. Preparing of needed sets
    



```python
train_size = int(df_spam.shape[0] * (1 - test_size - valid_size))
valid_bound = int(df_spam.shape[0] * (1 - valid_size))

train_sentences = sentences_new_set[0 : train_size]
valid_sentences = sentences_new_set[train_size : valid_bound]
test_sentences = sentences_new_set[valid_bound : ]

train_labels_str = labels_new_set[0 : train_size]
valid_labels_str = labels_new_set[train_size : valid_bound]
test_labels_str = labels_new_set[valid_bound : ]
```

### II stage. Labels transformations.


Secondly, we will replace all the labels (with the following values: `ham` and `spam`) to the appropriate values `1` and `0`, and transform them to Numpy arrays.





```python
print("Stage II. Labels transformations\n")

train_labels = [0] * len(train_labels_str)
for ind, item in enumerate(train_labels_str):
    if item == 'ham':
        train_labels[ind] = 1
    else:
        train_labels[ind] = 0
        
valid_labels = [0] * len(valid_labels_str)
for ind, item in enumerate(valid_labels_str):
    if item == 'ham':
        valid_labels[ind] = 1
    else:
        valid_labels[ind] = 0

test_labels = [0] * len(test_labels_str)
for ind, item in enumerate(test_labels_str):
    if item == 'ham':
        test_labels[ind] = 1
    else:
        test_labels[ind] = 0

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)
test_labels = np.array(test_labels)
```

    Stage II. Labels transformations
    


### III stage. Tokenization.


[Tokenization](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01) is a process of splitting up a large body of text into smaller lines or words. It helps in interpreting the meaning of the text by analyzing the sequence of the words. We converted our output feature into a numerical form, then, what about the input feature based on `size_vocabulary`.

First, let’s tokenize our data and convert it into a numerical sequence using `Keras` `Tokenizer`. We can also find the index number `word_index` of the corresponding words. We will need a really big word index to handle sentences that are not in the training set. This can be handled using the `Out Of Vocabulary` <OOV> token variable `oov_token`.





```python
print("Stage III. Tokenization\n")

tokenizer = Tokenizer(num_words = size_vocabulary,
                      oov_token = oov_token,
                      lower = False)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
```

    Stage III. Tokenization
    


As you can see in `text_to_sequence`, all the sequences are of different lengths which are not compatible for the model to train. So we should make all the sentences length equal. For this, we are padding the sequences with `padding_type`.



```python
train_sequences = tokenizer.texts_to_sequences(train_sentences)
size_voc = len(word_index) + 1
max_len = max([len(i) for i in train_sequences])
train_set = pad_sequences(train_sequences,
                                padding = padding_type,
                                maxlen = max_len,
                                truncating = trunc_type) 

valid_sequences = tokenizer.texts_to_sequences(valid_sentences)
valid_set = pad_sequences(valid_sequences,
                               padding = padding_type,
                               maxlen = max_len,
                               truncating = trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_set = pad_sequences(test_sequences,
                               padding = padding_type,
                               maxlen = max_len,
                               truncating = trunc_type)
```

### IV stage. Model building.


You should create your own model at this stage.

The first layer of the model is `Embedding` layer, which can be used to create `dense` encoding of words based on an input `size_voc` of defined vocabulary (in our case it's the index number `word_index` of the corresponding words `+ 1`). Typically sparse and dense word encodings denote coding efficiency.

Further, we use one (you can change this number) pair of layers: `Dense` & `Dropout`. You can choose a number of layers pairs yourself.

Using `bidirectional LSTM` will run your input in two ways: one from the past to the future and one from the future to the past (in a back way). This distinguishes this approach from `unidirectional LSTM` which works in the opposite direction, so you save information from the future. Thus, by using the two hidden states together, you can save information from both the past and the future at any time.

`Dropout` [layer](https://arxiv.org/abs/1207.0580?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkQuickLabsAdvanced_ML_DL_spam_classification_L427910497-2022-01-01) is used in neural networks to solve the problem of overfitting. Networks for training are obtained by dropping out neurons with probability `p`, so the probability that a neuron will remain in the network is `1 - p`.

`Dense` layer is an ordinary tightly bonded layer of a neural network where each neuron is connected to all inputs.



```python
print("Stage IV. Model building\n")

model = Sequential([
    Embedding(size_voc, embedding_dimension, input_length = max_len),
    Bidirectional(LSTM(100)),
    Dropout(drop_level),
    Dense(20, activation = 'relu'),
    Dropout(drop_level),
    Dense(1, activation = 'sigmoid')
])
```

    Stage IV. Model building
    


### V stage. Model compiling & fitting.


This stage allows you to train your model, but firstly, you should set some hyperparameters & other variables values, such as `batch size`, number of `epochs` for training, types of `optimizer` & `loss` function. You can change all or a part of them during your research.


```python
print("Stage V. Model compiling & fitting\n")
optim = Adam(learning_rate = 0.0001)

model.compile(loss = 'binary_crossentropy',
              optimizer =  optim,
              metrics = ['accuracy'])
model.summary()
```

    Stage V. Model compiling & fitting
    
    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 189, 64)           606080    
    _________________________________________________________________
    bidirectional_2 (Bidirection (None, 200)               132000    
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 200)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 20)                4020      
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 20)                0         
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 21        
    =================================================================
    Total params: 742,121
    Trainable params: 742,121
    Non-trainable params: 0
    _________________________________________________________________



```python
history = model.fit(train_set, 
                    train_labels,
                    epochs = num_epochs, 
                    validation_data = (valid_set, valid_labels),
                    workers = os.cpu_count(),
                    use_multiprocessing = True,
                    verbose = 1)
```

    2025-04-15 02:56:55.099694: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
    2025-04-15 02:56:55.100819: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 2494065000 Hz


    Epoch 1/20
    122/122 [==============================] - 103s 817ms/step - loss: 0.6030 - accuracy: 0.8127 - val_loss: 0.3260 - val_accuracy: 0.8803
    Epoch 2/20
    122/122 [==============================] - 79s 647ms/step - loss: 0.3183 - accuracy: 0.8712 - val_loss: 0.2647 - val_accuracy: 0.8803
    Epoch 3/20
    122/122 [==============================] - 79s 650ms/step - loss: 0.2481 - accuracy: 0.8825 - val_loss: 0.1998 - val_accuracy: 0.9421
    Epoch 4/20
    122/122 [==============================] - 78s 643ms/step - loss: 0.1697 - accuracy: 0.9550 - val_loss: 0.1536 - val_accuracy: 0.9498
    Epoch 5/20
    122/122 [==============================] - 78s 640ms/step - loss: 0.1212 - accuracy: 0.9652 - val_loss: 0.0984 - val_accuracy: 0.9653
    Epoch 6/20
    122/122 [==============================] - 78s 638ms/step - loss: 0.0863 - accuracy: 0.9746 - val_loss: 0.0725 - val_accuracy: 0.9846
    Epoch 7/20
    122/122 [==============================] - 80s 655ms/step - loss: 0.0705 - accuracy: 0.9835 - val_loss: 0.0700 - val_accuracy: 0.9846
    Epoch 8/20
    122/122 [==============================] - 78s 643ms/step - loss: 0.0567 - accuracy: 0.9844 - val_loss: 0.0602 - val_accuracy: 0.9884
    Epoch 9/20
    122/122 [==============================] - 82s 668ms/step - loss: 0.0429 - accuracy: 0.9876 - val_loss: 0.0604 - val_accuracy: 0.9884
    Epoch 10/20
    122/122 [==============================] - 78s 642ms/step - loss: 0.0370 - accuracy: 0.9896 - val_loss: 0.0587 - val_accuracy: 0.9923
    Epoch 11/20
    122/122 [==============================] - 79s 651ms/step - loss: 0.0394 - accuracy: 0.9910 - val_loss: 0.0642 - val_accuracy: 0.9884
    Epoch 12/20
    122/122 [==============================] - 76s 624ms/step - loss: 0.0459 - accuracy: 0.9895 - val_loss: 0.0635 - val_accuracy: 0.9884
    Epoch 13/20
    122/122 [==============================] - 76s 626ms/step - loss: 0.0322 - accuracy: 0.9915 - val_loss: 0.0620 - val_accuracy: 0.9923
    Epoch 14/20
    122/122 [==============================] - 75s 615ms/step - loss: 0.0293 - accuracy: 0.9947 - val_loss: 0.0692 - val_accuracy: 0.9846
    Epoch 15/20
    122/122 [==============================] - 75s 616ms/step - loss: 0.0334 - accuracy: 0.9934 - val_loss: 0.0609 - val_accuracy: 0.9923
    Epoch 16/20
    122/122 [==============================] - 74s 608ms/step - loss: 0.0456 - accuracy: 0.9881 - val_loss: 0.0618 - val_accuracy: 0.9923
    Epoch 17/20
    122/122 [==============================] - 74s 607ms/step - loss: 0.0270 - accuracy: 0.9951 - val_loss: 0.0666 - val_accuracy: 0.9923
    Epoch 18/20
    122/122 [==============================] - 74s 609ms/step - loss: 0.0177 - accuracy: 0.9957 - val_loss: 0.0675 - val_accuracy: 0.9923
    Epoch 19/20
    122/122 [==============================] - 74s 605ms/step - loss: 0.0223 - accuracy: 0.9951 - val_loss: 0.0705 - val_accuracy: 0.9884
    Epoch 20/20
    122/122 [==============================] - 78s 636ms/step - loss: 0.0188 - accuracy: 0.9962 - val_loss: 0.0744 - val_accuracy: 0.9846


### VI stage. Results visualization.



```python
print("Stage VI. Results visualization\n")
plot_history(history)
```

    Stage VI. Results visualization
    



    
![png](output_94_1.png)
    


If you can see the values reduction for the `loss` distribution, and if you see the values increase for the `accuracy`, then it's a good sign. It means your model training goes in the right direction.

Thus, the main goal has been reached.

In addition, let's estimate your pre-built model on the test set which this model hasn't seen in any case.



```python
model_score = model.evaluate(test_set, test_labels, batch_size = embedding_dimension, verbose = 1)
print(f"Test accuracy: {model_score[1] * 100:0.2f}% \t\t Test error: {model_score[0]:0.4f}")
```

    17/17 [==============================] - 2s 142ms/step - loss: 0.0664 - accuracy: 0.9836
    Test accuracy: 98.36% 		 Test error: 0.0664


### VII stage. Model saving & predict checking.


We can save our model and tokenizer for future uses in different formats. We have to do two more steps: to save our trained model so that we can use it in the further research. In addition, we should check our saved model and try to make a forecast.

You should enter any name of your model for saving but make sure that it is in quotes.



```python
M_name = "My_model"
pickle.dump(tokenizer, open(M_name + ".pkl", "wb"))
filepath = M_name + '.h5'
tf.keras.models.save_model(model, filepath, include_optimizer = True, save_format = 'h5', overwrite = True)
print("Size of the saved model :", os.stat(filepath).st_size, "bytes")
```

    Size of the saved model : 8961640 bytes


Let's find a set of predictions based on our model. We will enter the `threshold` value (`0.5`) which will help us to mark correctly and incorrectly predicted labels. In addition, we will build a `classification report` (as for the previous studied ML models - see `Part A`) and draw the `confusion matrix`.



```python
y_pred_bLSTM = model.predict(test_set)

y_prediction = [0] * y_pred_bLSTM.shape[0]
for ind, item in enumerate(y_pred_bLSTM):
    if item > threshold:
        y_prediction[ind] = 1
    else:
        y_prediction[ind] = 0

conf_m_bLSTM = confusion_matrix(test_labels, y_prediction)
class_rep_bLSTM = classification_report(test_labels, y_prediction)
print('\t\t\tClassification report:\n\n', class_rep_bLSTM, '\n')
plot_conf_matr(conf_m_bLSTM, classes = ['Spam','Ham'], normalize = False, title = 'Confusion matrix for bLSTM')
```

    			Classification report:
    
                   precision    recall  f1-score   support
    
               0       0.93      0.92      0.93       117
               1       0.99      0.99      0.99       917
    
       micro avg       0.98      0.98      0.98      1034
       macro avg       0.96      0.96      0.96      1034
    weighted avg       0.98      0.98      0.98      1034
     
    



    
![png](output_101_1.png)
    


Let's check our trained model on the real messages which you can create yourself.



```python
# You can change this message (as any short sentence) yourself
message_example = ["Darling, please give me a cup of tea"] 

message_example_tp = pad_sequences(tokenizer.texts_to_sequences(message_example),
                                   maxlen = max_len,
                                   padding = padding_type,
                                   truncating = trunc_type)

pred = float(model.predict(message_example_tp))
if (pred > threshold):
    print ("This message is a real text")
else:
    print("This message is a spam message")
```

    This message is a real text



```python

```

