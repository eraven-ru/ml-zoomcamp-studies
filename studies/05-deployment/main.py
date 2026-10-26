# Section: Deployment

# ii: plan -

# export model as a file
# create a web service for the model (flask)
# create an environment for python dependencies (pipenv)
# create an environment for system dependencies (docker)
# load into cloud (aws; optional)

# Section: Exporting model as a file

# save the code in train.py that uses pickle to save the model in a binary file
# create predict.py that uses the binary to predict the probability of churning for a user object

# Section: Web Service

# using flask create a ping-pong app (DONE)
# with flask create a WS accepting a POST request with a customer & returning churn results (DONE)
# imitate POST request with requests library (DONE)
# use waitress to launch a WS (WSGI interface)

# Section: Virtual Environment

# creating a separate virtual env allows to have a separate defined set of libraries for the project
# -- separate needed python, pip & packages with needed versions only - to be able to recreate the environment

# different packages allow that: venv, conda, poetry, we will use pipenv
# install globally & use for installation like pip command
# then run commands inside environment's shell with "pipenv shell" or "pipenv run ..."