from invoke import task

"""Run, for example:

$ invoke run-unit-tests
"""

@task
def tests(c):
    c.run("python3 -b -m pytest -s -l -v --rootdir=tests/")
