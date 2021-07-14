Free copyright filter to fulfill EU Copyright Directive

### Install dependencies

The python client depends on `opencv-contrib-python` or `opencv-contrib-python-headless`

```
git clone https://github.com/framespot/client-py.git
cd client-py
pip install -r requirements.txt
```

### Inference copyright

```
python . --verbose /path/to/movie.mp4
python . --verbose /path/to/stockphoto.jpg
```

### Example result

```JSON
[{
  "uri": "https://www.imdb.com/title/tt2380307/",
  "ids": {"imdb": "tt2380307", "tmdb": "movie/354912"},
  "title": "Coco",
  "year": "2017",
  "genres": ["Animation","Family","Comedy","Adventure","Fantasy"],
  "companies": ["Pixar","Walt Disney Pictures"],
  "homepage": "https://www.pixar.com/feature-films/coco",
  "poster": "https://www.themoviedb.org/t/p/original/gGEsBPAijhVUFoiNpgZXqRVWJt2.jpg",
  "frames": [{
    "type": "movie",
    "season": null,
    "episode": null,
    "offset": 1855,
    "hamming": 8,
    "matrix": [[ 1.001, 0.008,-0.001],
               [-0.008, 1.001, 0.004]]
  }]
}]
```
