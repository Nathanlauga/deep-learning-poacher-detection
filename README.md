Deep Learning project : Poacher detection on drone images
-----

This GitHub is a repository for a school project at `Ynov Bordeaux`.

Team's member :
- Paul Colinmaire
- Laetitia Constantin
- Maxime Tilhet
- Nathan Lauga

# Setup environment

For this project we use tensorflow v2.1.0.

You can use 4 possibilities to install the environment : 
1. With docker-compose and tensorflow GPU
2. With docker-compose and tensorflow no GPU
3. With Python and tensorflow GPU
4. With Python and tensorflow no GPU

### 1. docker-compose and tensorflow GPU

Please checkout requirements [GPU Support | Tensorflow](https://www.tensorflow.org/install/gpu).

```bash
cd docker_gpu
docker-compose build
docker-compose up
```
### 2. docker-compose and tensorflow no GPU

```bash
cd docker_no_gpu
docker-compose build
docker-compose up
```

### 3. With Python and tensorflow GPU

Use Python 3.6[+] and checkout requirements [GPU Support | Tensorflow](https://www.tensorflow.org/install/gpu).

```bash
pip install -r requirements_gpu.txt
```

### 4. With Python and tensorflow no GPU

Use Python 3.6[+]

```bash
pip install -r requirements_no_gpu.txt
```
