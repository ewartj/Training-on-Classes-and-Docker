# Dockerisation 👀
Community of Practice Session 2 with UCLH

By Matt Stammers, Catalina Carenzo and Jonny Sheldon

# Session:

Data Courtesy of: https://www.kaggle.com/datasets/joniarroba/noshowappointments thanks to https://www.kaggle.com/joniarroba

This takes the models found in eda_explosion from [Python_Training_for_BI_Analysts](https://github.com/MattStammers/Python_Training_For_BI_Analysts) and wraps it into Python classes

## Docker

To run:

```sh
docker build -t docker_example:test .
docker run --name docker -v /home/shelde2/Training-on-Classes-and-Docker/no_flask/04_Containerised_App/static/data:/app/app/static/data -it docker_example:test /bin/bash
```

To kill all the containers and start again

```sh
docker stop $(docker ps -aq)
```

Then run the container again to start it up once more.

## Enjoy! 😁

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
