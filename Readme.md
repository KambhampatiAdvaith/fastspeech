# FastSpeech 2 Text-to-Speech for English and Telugu

A multilingual Text-to-Speech (TTS) system built using **FastSpeech 2**, with separate acoustic models for English and Telugu. The project uses language-specific text processing pipelines and a pretrained universal HiFi-GAN vocoder to generate speech from text.

## Project Overview

This project implements two independent FastSpeech 2 models:

- **English:** Trained using the Svarah dataset from AI4Bharat.
- **Telugu:** Trained using the IndicTTS Telugu dataset.

The implementation is based on the PyTorch FastSpeech 2 repository by [ming024](https://github.com/ming024/FastSpeech2), with modifications for language-specific preprocessing, text-to-phoneme conversion, training, and inference.

## Model Details

| Component | English | Telugu |
|---|---|---|
| Dataset | Svarah | IndicTTS Telugu |
| Training steps | 24,000 | 20,000 |
| Training audio | 4.86 hours | 4.94 hours |
| Speakers | 91 | 2 |
| Acoustic model | FastSpeech 2 | FastSpeech 2 |
| Vocoder | Universal HiFi-GAN | Universal HiFi-GAN |

Both models use the same FastSpeech 2 architecture, with language-specific speaker embeddings and text front ends.

## Architecture

The system consists of the following components:

1. **Text Front End:** Normalizes input text and converts it into phoneme sequences.
2. **Encoder:** Processes phoneme sequences using Feed-Forward Transformer blocks.
3. **Variance Adaptor:** Predicts duration, pitch, and energy.
4. **Length Regulator:** Expands phoneme representations into frame-level representations.
5. **Mel Decoder and PostNet:** Generates refined mel-spectrograms.
6. **HiFi-GAN Vocoder:** Converts mel-spectrograms into speech waveforms.

## Datasets

### English – Svarah
- Source: AI4Bharat Svarah.
- Contains Indian-accented English speech.
- Training data after preprocessing and alignment: 4.86 hours.
- 91 speakers were selected.

Dataset: https://huggingface.co/datasets/ai4bharat/Svarah

### Telugu – IndicTTS
- Source: SPRINGLab IndicTTS Telugu.
- Contains studio-recorded Telugu speech.
- Training data after preprocessing and alignment: 4.94 hours.
- Two speakers were used.

Dataset: https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu

## Technologies Used

- Python
- PyTorch
- FastSpeech 2
- HiFi-GAN
- Montreal Forced Aligner (MFA)
- Librosa
- PyWorld
- g2p_en
- NumPy and SciPy
- Matplotlib

## Repository Structure

```text
FastSpeech2-TTS/
│
├── English/
│   ├── config.yaml
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
│
├── Telugu/
│   ├── config.yaml
│   ├── preprocess.py
│   ├── train.py
│   └── inference.py
│
├── text/
│   ├── telugu.py
│   ├── english_frontend.py
│   └── symbols.py
│
├── scripts/
│   ├── setup_env.sh
│   ├── run_mfa.sh
│   ├── export_checkpoint.py
│   └── plot_losses.py
│
├── Results/
│   ├── English/
│   └── Telugu/
│
├── requirements.txt
└── README.md
```

*The structure above highlights the main project files; other modules from the FastSpeech 2 implementation are also required.*

## Installation

Clone the repository:

```bash
git clone https://github.com/KambhampatiAdvaith/fastspeech
cd FastSpeech2-TTS
```

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Run the environment setup script:

```bash
bash scripts/setup_env.sh
```

Montreal Forced Aligner must be installed separately using conda-forge.

## Preprocessing and Training

### English

```bash
python English/preprocess.py select
bash scripts/run_mfa.sh English
python English/preprocess.py features
python English/train.py
```

Set `HF_TOKEN` before selecting the English dataset.

### Telugu

```bash
python Telugu/preprocess.py select
bash scripts/run_mfa.sh Telugu
python Telugu/preprocess.py features
python Telugu/train.py
```

To resume training from a checkpoint, use the `--restore_step` argument supported by the training scripts.

## Inference

Generate English speech:

```bash
python English/inference.py --restore_step 24000 --controls
```

Generate Telugu speech:

```bash
python Telugu/inference.py --restore_step 20000 --controls
```

The inference pipeline generates speech and can also produce duration, pitch, and energy visualizations and control experiments.

## Model Checkpoints

The trained synthesis checkpoints are:

| Language | Training checkpoint |
|---|---|
| English | Step 24,000 |
| Telugu | Step 20,000 |

## Model Checkpoints

The English and Telugu FastSpeech 2 models were trained separately using their respective datasets.

The trained model checkpoints are maintained privately and are not included in this public repository.

## Results

The project includes:

- Generated English and Telugu speech samples.
- Training and validation loss plots.
- Mel-spectrogram visualizations.
- Predicted duration, pitch (F0), and energy plots.
- `predictions.json` files.
- Speed, pitch, and energy control experiments.

Results are organized under `Results/English` and `Results/Telugu`.

## Limitations

- Both models show signs of overfitting during later training stages.
- The universal HiFi-GAN vocoder was not fine-tuned on the target datasets.
- Some generated utterances contain audible artefacts.
- Telugu text processing uses a rule-based phoneme conversion system that has not been validated by a native-speaker evaluation.
- Evaluation was informal and did not include objective speech-quality metrics.

## References

1. Ren et al. (2021), [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558).
2. [FastSpeech 2 – PyTorch Implementation](https://github.com/ming024/FastSpeech2).
3. Kong et al. (2020), [HiFi-GAN](https://arxiv.org/abs/2010.05646).
4. [Montreal Forced Aligner](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html).
5. [AI4Bharat Svarah Dataset](https://github.com/AI4Bharat/Svarah).
6. [IIT Madras TTS and Indic Resources](https://indicnlp.ai4bharat.org/resources/).

## Author

**Kambhampati Advaith**
