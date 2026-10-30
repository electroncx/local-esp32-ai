# local-esp32-ai

A 5.5M-parameter offline survival and everyday assistant that runs entirely on an ESP32.
No Wi-Fi, no cloud, no signal needed. It answers directly: the answer, nothing else.

> **It might make a mistake.** Double-check anything that matters. It is not a substitute for first-aid or survival training: **in an emergency, call for help first** if you can.

```
you: how do i purify water
esp: Boil it for 1 minute (3 above 5,000 ft).
you: what do i do if i see a bear
esp: Stay calm, don't run; talk and back away slowly.
you: signs of hypothermia
esp: Shivering, confusion, slurred speech, drowsy.
you: thunder 10 seconds after lightning
esp: About 2.1 miles (3.4 km). Go indoors.
you: how much does a hummingbird weigh
esp: Between 2 and 20 grams.
you: how do i build a raft
esp: I don't know.
```

## Specs

| | |
|---|---|
| Model | 5,472,640-parameter GPT: 6 layers, dim 320, 5 heads, context 96 |
| Weights | 2-bit (fp16 scale per 32 weights), quantization-aware trained |
| Knowledge | 37,288 facts (288 survival), 146,994 accepted phrasings |
| Model file | 3.5 MB, filling a 4 MB board's flash |
| RAM | about 90 KB, plus the spare heap (about 165 KB) put to work for speed: questions run through the model in 17-character blocks |
| Boards | ESP32, ESP32-S3, ESP32-C3 with 4 MB flash |
| Answers | 48 characters max, or `I don't know.` |


## What it knows

- **Survival:** 288 facts on water, fire, shelter, signaling, navigation, food, cold, heat, wildlife, weather, first aid and knots, from CDC, NWS, National Park Service, Ready.gov, Red Cross, NHS, WHO and the US Army survival manual.
- **General knowledge:** 33,712 facts from Wikidata (people, places, books, films, science) and about 3,300 more from datasets and hand-written lists.
- **Calculated, not memorized:** math, money, units, times, dates, water for N people, bleach doses, lightning distance.

Anything it doesn't know gets `I don't know.` instead of a guess.

## Usage

**First time (once, with a computer):**

1. Download this project: on GitHub click **Code → Download ZIP** and unzip it. No GitHub account needed.
2. Install the [Arduino IDE](https://www.arduino.cc/en/software). In **Tools → Board → Boards Manager**, search `esp32` and install **esp32 by Espressif Systems** (version 3.x).
3. In the Arduino IDE, **File → Open** the sketch `arduino/local_esp32_ai/local_esp32_ai.ino` from the unzipped folder.
4. Set the **Tools** menu:
   - **Board:** ESP32 Dev Module (ESP32-S3 Dev Module or ESP32C3 Dev Module for those chips)
   - **Partition Scheme:** **Custom** (the AI needs almost all of the 4 MB flash; the sketch brings its own partition table)
   - **Flash Size:** 4MB
   - **Port:** your ESP32, after plugging it in with a USB cable
5. Click **Upload** (→). The first build takes a few minutes (the AI is 3.6 MB); then it copies onto the ESP32. If the board doesn't start after uploading, set **Tools → Flash Mode** to **DIO** and upload again.

**Every time after that:**

1. Plug the ESP32 into any computer (or a phone with a USB serial app). No internet, no GitHub, no uploading again: the AI lives on the ESP32.
2. Open **Tools → Serial Monitor** in the Arduino IDE (or any serial terminal) at **115200 baud**, line ending **Newline**. The terminal is just a keyboard and screen for the ESP32.
3. Wait for `tinyai ready`, type a question, press Enter.
4. **Keep it short.** `hypothermia` works; `i think my friend has hypothermia what do i do` doesn't.
5. `I don't know.` means no matching fact: try fewer words, or it doesn't know.

```
you: purify water
esp: Boil it for 1 minute (3 above 5,000 ft).
```

More examples and tips: [docs/usage.md](docs/usage.md). To try it on a PC without an ESP32: `make -C host && ./host/tinyai firmware/data/model.bin`.

## Results

| | |
|---|---|
| Trained questions answered exactly | 146,995 / 146,995 |
| Test questions it never trained on | 399 / 417 right, **0 wrong** (the other 18 say `I don't know.`) |
| Real firmware in the ESP32 emulator | same answer as the PC on 350 / 350 |

On survival questions worded in ways it hasn't seen, expect it to answer about half and refuse the rest. Short, plain questions work best. Run every check yourself with `./test.sh` (details in [docs/details.md](docs/details.md#results)).

## Train your own

```sh
pip install -r requirements.txt
# the shipped model: about 8 hours on a 4-core CPU, no GPU
python train/train.py --tier max4mb --weights int2 --qat-from 0.5 --steps 15000 \
    --dim 320 --layers 6 --heads 5 --kv-heads 1 --hidden 1024 --out train/ckpt_2.pt
make -C host && python train/export.py --ckpt train/ckpt_2.pt --max-bytes 3756000
python firmware/make_arduino.py    # refresh the Arduino IDE sketch
./test.sh
```

`python train/train.py` alone trains the smaller 2M 4-bit model (24,855 facts, about 3 hours). To add facts, add a line to `data/facts.tsv` or `data/facts_survival.tsv` and retrain:

```
question|another phrasing<TAB>Direct answer.
```

Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) curriculum.

How it works, how it was tested, and its limits: [docs/details.md](docs/details.md).
