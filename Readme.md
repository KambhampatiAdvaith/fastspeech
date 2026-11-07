# FastSpeech 2 Text-to-Speech for English and Telugu

Two separate FastSpeech 2 models, one per language, trained on about 5 hours of audio each (plus about 1 hour for validation):

| | English | Telugu |
|---|---|---|
| Dataset | Svarah (AI4Bharat), 91 speakers | IndicTTS Telugu (Hugging Face copy), 2 speakers |
| Training audio / validation audio | 4.86 h / 1.08 h | 4.94 h / 1.00 h |
| Training steps | 24,000 | 20,000 |
| Best validation loss | 3.530 (step 3,000) | 3.590 (step 2,000) |
| Validation loss at the final step | 3.712 | 4.067 |
| Checkpoint in this repo | `English/checkpoints/24000.pth.tar` | `Telugu/checkpoints/20000.pth.tar` |

Both models overfit early, and the final checkpoints (not the validation minimum) are the ones used for the generated audio. The full analysis is in the technical report: **`Report/Technical_Report.pdf`**.

The code is based on the PyTorch implementation [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) of the paper *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (Ren et al., ICLR 2021). The pretrained universal HiFi-GAN vocoder from that repository is used as is and was not retrained.

## Repository layout

```
English/                  preprocess.py  train.py  inference.py  config.yaml  test_sentences.json  checkpoints/
Telugu/                   preprocess.py  train.py  inference.py  config.yaml  test_sentences.json  checkpoints/
fastspeech2/              reference code with our modifications (model, text front ends, audio, utils)
scripts/                  data selection, forced alignment, loss plots, checkpoint export, setup
preprocessed_data/        small files needed for inference (stats.json, speakers.json, train.txt, val.txt)
Results/
  English/ and Telugu/    training_loss.png, validation_loss.png, test_generated_audios/,
                          variance_adaptor_plots/{duration,pitch,energy}/, spectrograms/,
                          predictions.json, control_experiments/
Report/Technical_Report.pdf
requirements.txt
```

The large preprocessed features (mel, pitch, energy, duration), the raw audio and the alignments are not in the repository. They are rebuilt by the preprocessing steps below.

## Quick start: synthesise the test sentences with the included checkpoints

The repository contains the final checkpoints (weights only, fp16, about 70 MB each), so inference can be run without retraining. A CPU is enough for the five sentences. Run from the repository root:

```bash
bash scripts/setup_env.sh          # installs packages, NLTK data and the pretrained HiFi-GAN vocoder, and a Telugu font
python English/inference.py --restore_step 24000 --controls
python Telugu/inference.py  --restore_step 20000 --controls
```

Outputs are written to `Results/English` and `Results/Telugu`: the audio for every test sentence, the duration, pitch (F0) and energy plots, spectrograms, a `predictions.json` file with the predicted values, and, with `--controls`, speed, pitch and energy variations. Add `--speaker NAME` to choose a speaker in the multi-speaker models (the default is the speaker with the most training data: `spk000` for English, the speaker with code 1 for Telugu).

Inference needs `stats.json`, `speakers.json` and `train.txt` from `preprocessed_data/<Language>/`, which are included.

## Test sentences

The sentences are stored in `English/test_sentences.json` and `Telugu/test_sentences.json` with the IDs E1, E2a, E2b, E3, E4 and T1, T2a, T2b, T3, T4. The text is kept as given in the assignment, including the stray space in E3 ("rhyth m").

## Environment

Developed and run on Google Colab (preprocessing, English training up to step 12,000) and Kaggle (English training from step 12,000, Telugu training), with Python 3.13 and the PyTorch version preinstalled on those runtimes (the exact version was not recorded; run `pip freeze` to record yours).

```bash
pip install -r requirements.txt
```

[Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/) is needed for preprocessing only and is installed with conda (`conda install -c conda-forge montreal-forced-aligner`). Its command-line flags change between versions, so check `mfa align --help` if a flag is rejected.

## Reproducing the full pipeline

Run everything from the repository root.

**English (Svarah).** The dataset is gated on Hugging Face: accept the terms at https://huggingface.co/datasets/ai4bharat/Svarah and set a read token.

```bash
export HF_TOKEN=...                          # your Hugging Face token
python English/preprocess.py select --max_speakers 115 --max_dur 15
bash   scripts/run_mfa.sh English            # forced alignment
python English/preprocess.py features        # mel, pitch, energy, durations, train/val split
python English/train.py                      # resume with: --restore_step N
python scripts/plot_losses.py English
python English/inference.py --restore_step 24000 --controls
```

**Telugu (IndicTTS, Hugging Face copy `SPRINGLab/IndicTTS_Telugu`).**

```bash
python Telugu/preprocess.py select
bash   scripts/run_mfa.sh Telugu             # trains an alignment model on the 6 h, then aligns
python Telugu/preprocess.py features
python Telugu/train.py
python scripts/plot_losses.py Telugu
python Telugu/inference.py --restore_step 20000 --controls
```

The training length is set in each `config.yaml` (`total_step`, `anneal_steps`). The values used are 24,000 and 20,000.

## Data selection and split

Both languages use the same procedure with random seed 13: clips are filtered by length and transcript quality, about 6 hours are sampled at random, and the sample is split by utterance into about 5 hours of training audio and 1 hour of validation audio, stratified by speaker. The split is written to a manifest and checked in code, so no utterance is in both sets. Validation shares speakers with training.

- **English:** 4,752 of 6,656 Svarah utterances kept (1.5 to 15 s, plain-text transcript). The Hugging Face release has no speaker-id column, so speakers are identified by the combination of the speaker-metadata columns (115 keys found; Svarah documents 117 speakers). After alignment: 3,026 training and 645 validation utterances.
- **Telugu:** about 3 hours drawn from each of the two speakers in the dataset (neither has 6 hours alone), so the Telugu model is a two-speaker model. After alignment: 2,143 training and 438 validation utterances.

## Main changes to the reference implementation

| Area | Change |
|---|---|
| Compatibility | Current librosa and PyTorch (keyword arguments, `weights_only=False`), no hard-coded `.cuda()` calls, energy targets cast to float32 |
| Preprocessing | Only speaker directories are traversed, and files without an alignment are skipped |
| Bug fix | The energy embedding used the pitch control value; it now uses the energy control value |
| English front end | Number, date and time expansion, spelled-out AM and PM, and sentence-level G2P so noun and verb pronunciations of "conduct" are chosen by part of speech |
| Telugu front end (new) | Unicode normalisation, number, ordinal and time expansion into Telugu words, and a rule-based Telugu-script-to-phone converter; Telugu phones added to the symbol table |
| Configuration | One `config.yaml` per language; explicit train/validation split with an overlap check |
| Inference | New script that synthesises the test sentences and writes the duration, pitch and energy plots and the predicted values |

Details and the reasons for each change are in section 1 of the report.

## Checkpoints

Only the final checkpoint of each language is included. They are weights-only fp16 files exported with `scripts/export_checkpoint.py` so that they fit GitHub's file-size limit. Intermediate checkpoints saved during training (which include optimiser state and are several hundred MB each) are not included. The files are named by training step because the inference script loads `<restore_step>.pth.tar`.

## Known limitations

- Both models overfit early; the validation minimum is at step 3,000 (English) and step 2,000 (Telugu), and earlier checkpoints were not compared.
- The English config uses `max_seq_len: 1000` (11.6 s), so about 5% of the English training clips are only partly used.
- Predicted energy and F0 are not constrained, so occasional impossible values (for example slightly negative energy) occur.
- The Telugu number-word tables and G2P rules have not been checked by a native speaker.
- The evaluation is informal (one listener, no objective metrics). See sections 4 to 6 of the report.
- Training seeds are not fixed in the reference code, so a rerun will differ slightly.

## Licences and credits

- FastSpeech 2 code: [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence; the licence file is kept in `fastspeech2/LICENSE`).
- HiFi-GAN vocoder: Kong et al., *HiFi-GAN*, NeurIPS 2020.
- Forced alignment: Montreal Forced Aligner (McAuliffe et al., Interspeech 2017).
- Data: [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) (AI4Bharat) and the [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu) data derived from the [IIT Madras IndicTTS database](https://www.iitm.ac.in/donlab/indictts/database). Please follow the terms of use on those pages. The datasets are not redistributed here.

