# Classify-Falls-ML
This repo is for Queen's University ELEC 498 2018-2019 Group 2 Project. We seek to classify different movements gotten from smartphone sensors as either falling or not falling. We hope to use Python and Tensorflow intitally, then eventually code our own back-propagation algorithm to replace the Tenserflow libraries.

# Setup
1. Download Github Desktop
2. Sign in to Github Desktop (or Create an Account)
3. Download Python version 3.6.7 (https://www.python.org/downloads/release/python-367/) (because TensorFlow is not compatible with the latest version, 3.7), Isabel did the 'Windows x86-64 executable installer'
4. Download VSCode and Python plugin (the latter done by clicking on bottom left corner gear, 'Extensions')
5. Once Isabel has added you to the repo as collaborator, you can clone the repo onto your local machine.
6. To run the python file, open up command line and do: 
`cd C:\your-repo-path`
(To find your-repo-path, once have gotten repo cloned on GH Desktop, click Repository->'Show in Explorer', and copy that path)
`python helloworld2.py`
 7. Navigate to path where \Github is in command prompt, then create a virtual environment by doing `.\venv\Scripts\activate`
 8. Install tensorflow IN virtual environment by doing `pip tensorflow`(only have to do this installation of TF once, and then every time want to run a TF python file, just create a virtual environment again).

Isabel notes November 9, 2018:
- to run Python interpreter, have to do: `C:\Users\"Isabel Cumming"\AppData\Local\Programs\Python\Python36\python.exe`in command prompt. NOTE THE DOUBLE QUOTES AROUND MY NAME B/C OF SPACES (otherwise Command prompt will yell). 
- moved Python so that would be in path C:\Python\python.exe so that when run `python helloworld2.py` in the proper folder, it might see it. 
-uninstalled Python, then reinstalled it with 'Custom Installation' option, and put the Python36 folder directly in my C:/
- https://www.pythoncentral.io/add-python-to-path-python-is-not-recognized-as-an-internal-or-external-command/ (<---did that and still didnt work)



