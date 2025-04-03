import time

import torch
import torchaudio

from zonos.model import Zonos

texts = [
    "Rain lashed against the attic window.",
    "Dust motes danced in the single moonbeam slicing the darkness.",
    "A floorboard creaked downstairs.",
    "She held her breath, listening.",
    "Silence answered, heavy and absolute.",
    "Slowly, she lifted the rusted latch on the old trunk.",
    "A faint scent of lavender and forgotten years drifted out.",
    "Inside, nestled on velvet, lay a single, tarnished silver key.",
]
# Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral
emotions = [
    [1.0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1.0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1.0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1.0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1.0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1.0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1.0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1.0],
]
pitch_stds = range(80, 161, 20)


def main():
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model (here we use the transformer variant).
    print("Loading model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    model.requires_grad_(False).eval()

    # Load a reference speaker audio to generate a speaker embedding.
    print("Loading reference audio...")
    wav, sr = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)

    # Set a random seed for reproducibility.
    torch.manual_seed(421)

    def generator():
        for text, emotion, pitch_std in zip(texts, emotions, pitch_stds):
            yield {
                "text": text,
                "speaker": speaker,
                "emotion": emotion,
                "pitch_std": pitch_std,
                "language": "en-us",
            }

    # --- STREAMING GENERATION ---
    print("Starting streaming generation...")

    # Define chunk schedule: start with small chunks for faster initial output,
    # then gradually increase to larger chunks for fewer cuts
    stream_generator = model.stream(
        cond_dicts_generator=generator(),
        chunk_schedule=[17, *range(9, 100)],  # optimal schedule for RTX3090
        chunk_overlap=1,  # tokens to overlap between chunks (affects crossfade)
    )

    # Accumulate audio chunks as they are generated.
    audio_chunks = []
    t0 = time.time()
    generated = 0
    ttfb = None

    for i, audio_chunk in enumerate(stream_generator):
        audio_chunks.append(audio_chunk)
        elapsed = int((time.time() - t0) * 1000)
        if not i:
            ttfb = elapsed
        generated += int(audio_chunk.shape[1] / 44.1)
        print(f"Chunk {i + 1:>3}: elapsed {elapsed:>5}ms | generated up to {ttfb + generated:>5}ms")

    # Concatenate all audio chunks along the time axis.
    audio = torch.cat(audio_chunks, dim=-1).cpu()

    generation = round(time.time() - t0, 3)
    duration = round(audio.shape[1] / 44100, 3)

    print(f"TTFB: {ttfb}ms, generation: {generation}ms, duration: {duration}ms, RTX: {round(duration / generation, 2)}")

    # Save the full audio as a WAV file.
    out_sr = model.autoencoder.sampling_rate
    torchaudio.save("stream_sample.wav", audio, out_sr)
    print(f"Saved streaming audio to 'stream_sample.wav' (sampling rate: {out_sr} Hz).")

    # Or use the following to display the audio in the jupyter notebook:
    # from IPython.display import Audio
    # display(Audio(data=audio, rate=out_sr))


if __name__ == "__main__":
    main()
