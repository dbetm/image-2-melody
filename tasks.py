from invoke import task

"""Run, for example:

$ invoke run-unit-tests
"""

@task
def tests(c):
    c.run("python3 -b -m pytest -s -l -v --rootdir=tests/")


@task
def setup_experiment(c, exp: int):
    """Example: `invoke setup-experiment --exp 0`"""
    c.run(f"pip install --upgrade pip")
    c.run(f"pip install -r requirements-exp.txt")
    c.run(f"pip install -r experiments/exp_{exp}/requirements.txt")
