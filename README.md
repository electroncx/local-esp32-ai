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
| Knowledge | 33,788 facts answered (288 survival), 146,757 accepted phrasings |
| Model file | 3.6 MB, filling a 4 MB board's flash |
| RAM | about 90 KB (235 KB heap still free) |
| Boards | ESP32, ESP32-S3, ESP32-C3 with 4 MB flash |
| Answers | 48 characters max, or `I don't know.` |


## What it knows

- **Survival:** 288 facts on water, fire, shelter, signaling, navigation, food, cold, heat, wildlife, weather, first aid and knots, from CDC, NWS, National Park Service, Ready.gov, Red Cross, NHS, WHO and the US Army survival manual.
- **General knowledge:** 33,712 facts from Wikidata (people, places, books, films, science) and about 3,300 more from datasets and hand-written lists. The 3,500 lesser-known Wikidata facts the model gets wrong answer `I don't know.` instead.
- **Calculated, not memorized:** math, money, units, times, dates, water for N people, bleach doses, lightning distance.

Anything it doesn't know gets `I don't know.` instead of a guess.

## Usage

**First time (once, with a computer):**

1. Download this project: on GitHub click **Code → Download ZIP** and unzip it. No GitHub account needed.
2. Install [VS Code](https://code.visualstudio.com) and its **PlatformIO** extension.
3. Plug the ESP32 into the computer with a USB cable.
4. Open a terminal in the project's `firmware` folder and run `pio run -t upload`. This copies the AI onto the ESP32 (about a minute).

**Every time after that:**

1. Plug the ESP32 into any computer (or a phone with a USB serial app). No internet, no GitHub, no uploading again: the AI lives on the ESP32.
2. Open a serial terminal at **115200 baud**: `pio device monitor` in the `firmware` folder. The terminal is just a keyboard and screen for the ESP32.
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
| Trained questions answered exactly | 146,757 / 146,757 |
| Test questions it never trained on | 372 / 391 right, **0 wrong** (the other 19 say `I don't know.`) |
| Real firmware in the ESP32 emulator | same answer as the PC on 350 / 350 |

On survival questions worded in ways it hasn't seen, expect it to answer about half and refuse the rest. Short, plain questions work best. Run every check yourself with `./test.sh` (details in [docs/details.md](docs/details.md#results)).

## Train your own

```sh
pip install -r requirements.txt
# the shipped model: about 8 hours on a 4-core CPU, no GPU
python train/train.py --tier max4mb --weights int2 --qat-from 0.5 --steps 15000 \
    --dim 320 --layers 6 --heads 5 --kv-heads 1 --hidden 1024 --out train/ckpt_2.pt
make -C host && python train/export.py --ckpt train/ckpt_2.pt --max-bytes 3756000
./test.sh
```

`python train/train.py` alone trains the smaller 2M 4-bit model (24,855 facts, about 3 hours). To add facts, add a line to `data/facts.tsv` or `data/facts_survival.tsv` and retrain:

```
question|another phrasing<TAB>Direct answer.
```

Built by following the [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) curriculum.

How it works, how it was tested, and its limits: [docs/details.md](docs/details.md).
