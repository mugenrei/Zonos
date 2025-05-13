import gradio as gr
from gradio_interface import generate_audio

def main():
    # Define a very kawaii idol speech
    kawaii_speech = (
        "Hey everyone! I'm Crystal, the sparkly sweet idol here to bring extra sunshine to your day! "
        "Thank you so much for coming to see me and supporting my dream. "
        "Let's shine bright together and share our smiles with everyone!"
    )

    # Call generate_audio with the requested parameters
    generate_audio(
        model_choice="Zyphra/Zonos-v0.1-transformer",  # or any other model you want to use
        text=kawaii_speech,
        language="en-us",
        speaker_audio="./assets/test.mp3",
        prefix_audio="./assets/silence_100ms.wav",
        e1=1.00,
        e2=0.05,
        e3=0.05,
        e4=0.05,
        e5=0.05,
        e6=0.05,
        e7=0.10,
        e8=0.10,
        vq_single=0.5,
        fmax=24000,
        pitch_std=45,
        speaking_rate=15,
        dnsmos_ovrl=4.0,
        speaker_noised=False,
        cfg_scale=2,
        top_p=0,
        top_k=0,
        min_p=0,
        linear=0.5,
        confidence=0.4,
        quadratic=0,
        seed=6969,
        randomize_seed=False,
        unconditional_keys=[],
        use_windowing=True,
        progress=gr.Progress(),
    )

if __name__ == "__main__":
    main()