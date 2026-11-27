# birdcheck — i task di xkcd 1425 sono ora problemi semplicissimi

Questa repo è dedicata al fatto che la battuta in [xkcd 1425: "Tasks"](https://xkcd.com/1425/)
(24 settembre 2014) di Randall Munroe è invecchiata male.

Nella striscia una programmatrice dice che controllare se una foto è stata scattata in un
parco nazionale è una "semplice GIS lookup" che le basta qualche ora, ma controllare se la
foto ritrae un uccello richiederebbe «un team di ricerca e cinque anni». Nel title text Munroe
ricorda il progetto estivo del 1966 di Marvin Minsky: programmare un computer per riconoscere
oggetti in una scena, un problema che mezzo secolo dopo era ancora aperto.

Oggi entrambi i task della striscia sono problemi molto semplici:

- «la foto è di un uccello?» basta un modello pre-addestrato (MobileNetV2 su ImageNet, ~14 MB)
  e poche righe di Python;
- «la foto è stata scattata in un parco nazionale?» basta una GIS lookup: si leggono le
  coordinate GPS dai metadati EXIF della foto (se presenti) e si fa point-in-polygon sui
  confini dei parchi nazionali USA (dati U.S. National Park Service, bundled e semplificati
  in `src/birdcheck/national_parks.geojson`).

## Cosa fa

`birdcheck` prende una o più immagini e per ognuna risponde:

1. se la foto ritrae un uccello, classificandola con MobileNetV2 (ONNX, CPU) e controllando
   se la classe con probabilità massima è una delle classi ImageNet di uccelli;
2. se i metadati lo permettono (coordinate GPS EXIF presenti), se il punto di scatto cade
   dentro un parco nazionale; se la foto non ha GPS il check GIS risulta `unknown`
   (`--no-gis` lo disattiva).

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
birdcheck --no-gis foto.jpg
python -m birdcheck foto.jpg
```

Esempio di output:

```
foto.jpg: BIRD (robin, 99.3%); park: Yellowstone National Park (44.46050, -110.82830)
```

Possibili suffissi GIS: `park: <nome parco> (lat, lon)`, `park: outside national parks (lat, lon)`,
`park: unknown (no GPS metadata)`.

Exit code: `0` se almeno un'immagine ritrae un uccello, `1` se nessuna lo fa, `2` in caso di
errore di utilizzo.

Con `--json` viene stampato un oggetto JSON per riga con i campi della classificazione
(`path`, `is_bird`, `label`, `score`, `bird_score`, `top`) più i campi GIS (`gps`, `park`,
`in_national_park`; `null` quando i metadati GPS mancano).

## Uso come libreria

```python
from birdcheck import BirdClassifier, check_gis

prediction = BirdClassifier().predict("foto.jpg")
print(prediction.is_bird, prediction.label, prediction.score)

gis = check_gis("foto.jpg")
print(gis.in_national_park, gis.park, gis.lat, gis.lon)
```

## Dati GIS

`src/birdcheck/national_parks.geojson` contiene i confini dei parchi nazionali USA
(62 unità, UNIT_TYPE='National Parks') scaricati dal feature service ufficiale NPS
(`NPS_Land_Resources_Division_Boundary_and_Tract_Data_Service`, layer `nps_boundary`,
EPSG:4326) e semplificati con Douglas-Peucker (tolleranza ~0.003°, ~300 m). Il point-in-polygon
è implementato in Python puro (ray casting), senza dipendenze GIS esterne.

## Test

```bash
python -m unittest discover -s tests -v
```

I test di integrazione con il modello ONNX reale vengono saltati automaticamente se il file
non è presente in cache.

## Struttura

```
src/birdcheck/
  classifier.py            preprocess ImageNet, sessione ONNX, predizione uccello
  gis.py                   GPS EXIF + point-in-polygon sui parchi nazionali
  cli.py                   interfaccia a riga di comando
  labels.py                classi ImageNet di uccelli e caricamento etichette
  imagenet_labels.txt      le 1000 etichette ImageNet
  national_parks.geojson   confini dei parchi nazionali USA (semplificati)
tests/                     test unitari (stdlib unittest)
```
