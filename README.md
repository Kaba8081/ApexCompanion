# Apex Companion
This Python project tracks player death and down positions in [Apex Legends](https://store.steampowered.com/app/1172470/Apex_Legends/) matches and generates a detailed heatmap on a full-scale map. By analyzing past games, players can identify high-risk areas, improve their gameplay strategies, and learn from their positioning mistakes.

## Getting stared
### Prerequisites
1. Python 3.12.5
2. Tesseract
### Installation
1. Download the latest tesseract installer from [here](https://github.com/UB-Mannheim/tesseract/wiki),
2. Install tesseract to a `tesseract` folder next to the `ApexCompanion.py`,
3. Install all python dependencies using the `requirements.txt` file (`pip install -r requirements.txt`)

## Usage
Currently **Apex Companion** can only be used using the command line like in the example below.
```sh
python ApexCompanion.py [options]
```

It has a few modes of operation which can be selected using the following command line arguments:
* `-t` or `--tracker`:
Tracks your game state and creates screenshots of your death / down positions.
Usage:
```sh
python ApexCompanion.py --tracker true
```
* `-s` or `--source`:
Creates screenshots of your death / down positions from a given video file.
Usage: 
```sh
python ApexCompanion.py --source [path_to_file]
```
* `-g` or `--generate`:
Generates a heatmap from the generated screenshots on the given map.
Usage:
```sh
python ApexCompanion.py --generate true
```
* `--debug`:
Enables debug mode for testing / outputting more information to the command line.
```sh
python ApexCompanion.py --debug true [options]
```