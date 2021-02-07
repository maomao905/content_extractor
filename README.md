### Content Extraction API
- Extract main content from HTML excluding boilerplate (advertisement, SNS links, sidebars etc..)
It uses the features below and create a classifier whether it's main content or not. The model used for training is XGBoost with manual data of 600 articles and 80,000 labels (I labeled each block whether it's main content or not, thus one article has many bocks, so it has many labels per article)  
※ The training data and the model file are not included in this repository.

#### features
- distance from title
  - if the block is closer to the title, it is more likely main text.
- text density
  - how many characters each block contains
  - reduce density if it contains links
  - higher text density indicates it is main content
  - lower text density indicates it is advertisement or external links
- HTML attributes and attribute names
  - e.g.) `<article>`/`<p>` tag is more relevant to main content than `<a>`/`<span>` tag
  - e.g.) `<div class="main-content">` attribute name is more relevant than `<div class="ad-banner">`

### Usage
```sh
curl localhost:5000/extract/body -X POST -d '{
  "html": "<html>...</html>",
}'
```
OK
```json
{
  "status": "OK",
  "content": "body....",
  "image_urls": [
    "/images/entry/..."
  ],
  "score": 0.66
}
```
NG
```json
{
  "status": "NG",
  "error": "error.."
}
```

### Create features
```
$ docker-compose exec app python manager.py -t feature -d <file>
```

### Train
```
$ docker-compose exec app python manager.py -t train -d <file>
```

### License
MIT
