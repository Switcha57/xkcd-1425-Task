# birdcheck — il task di xkcd 1425 è ora un problema semplicissimo

Questa repo è dedicata al fatto che la battuta in [xkcd 1425: "Tasks"](https://xkcd.com/1425/)
(24 settembre 2014) di Randall Munroe è invecchiata male.

Nella striscia una programmatrice dice che controllare se una foto è stata scattata in un
parco nazionale è una "semplice GIS lookup" che le basta qualche ora, ma controllare se la
foto ritrae un uccello richiederebbe «un team di ricerca e cinque anni». Nel title text Munroe
ricorda il progetto estivo del 1966 di Marvin Minsky: programmare un computer per riconoscere
oggetti in una scena, un problema che mezzo secolo dopo era ancora aperto.

Oggi quel secondo task — «la foto è di un uccello?» — è un problema molto semplice: basta un
modello pre-addestrato (MobileNetV2 su ImageNet, ~14 MB) e poche righe di Python.

## Cosa fa

`birdcheck` prende una o più immagini e risponde se la foto ritrae un uccello, classificandola
con MobileNetV2 (ONNX, CPU) e controllando se la classe con probabilità massima è una delle
classi ImageNet di uccelli.

## Installazione

```bash
pip install .
```

Il modello viene scaricato automaticamente al primo utilizzo in
`~/.cache/birdcheck/mobilenetv2-7.onnx` (oppure si indica un modello locale con `--model`
o con la variabile d'ambiente `BIRDCHECK_MODEL`).

## Uso

```bash
birdcheck foto.jpg
birdcheck foto1.jpg foto2.jpg --json
python -m birdcheck foto.jpg
```

Esempio di output:

```
foto.jpg: BIRD (robin, 99.8%)
```

Exit code: `0` se almeno un'immagine ritrae un uccello, `1` se nessuna lo fa, `2` in caso di
errore di utilizzo.

Con `--json` viene stampato un oggetto JSON per riga con `path`, `is_bird`, `label`, `score`,
`bird_score` (probabilità totale delle classi uccello) e le top-5 classi.

## Uso come libreria

```python
from birdcheck import BirdClassifier

prediction = BirdClassifier().predict("foto.jpg")
print(prediction.is_bird, prediction.label, prediction.score)
```

## Test

```bash
python -m unittest discover -s tests -v
```

I test di integrazione con il modello reale vengono saltati automaticamente se il file ONNX
non è presente in cache.

## Struttura

```
src/birdcheck/
  classifier.py        preprocess ImageNet, sessione ONNX, predizione
  cli.py               interfaccia a riga di comando
  labels.py            classi ImageNet di uccelli e caricamento etichette
  imagenet_labels.txt  le 1000 etichette ImageNet
tests/                 test unitari (stdlib unittest)
```
