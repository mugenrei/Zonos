import re
import torch
import torchaudio
import gradio as gr
import numpy as np
from os import getenv
import io
from pydub import AudioSegment
import numpy as np
import nltk
import math
import importlib
from typing import Tuple, List, Optional, Any
import torch.nn.functional as F
import time
import logging

from zonos.model import Zonos, DEFAULT_BACKBONE_CLS as ZonosBackbone
from zonos.conditioning import make_cond_dict, supported_language_codes
from zonos.utils import DEFAULT_DEVICE as device

CURRENT_MODEL_TYPE = None
CURRENT_MODEL = None

SPEAKER_EMBEDDING = None
SPEAKER_AUDIO_PATH = None


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("audio_generation")

def load_model_if_needed(model_choice: str):
    global CURRENT_MODEL_TYPE, CURRENT_MODEL
    if CURRENT_MODEL_TYPE != model_choice:
        if CURRENT_MODEL is not None:
            del CURRENT_MODEL
            torch.cuda.empty_cache()
        print(f"Loading {model_choice} model...")
        CURRENT_MODEL = Zonos.from_pretrained(model_choice, device=device)
        CURRENT_MODEL.requires_grad_(False).eval()
        CURRENT_MODEL_TYPE = model_choice
        print(f"{model_choice} model loaded successfully!")
    return CURRENT_MODEL


def update_ui(model_choice):
    """
    Dynamically show/hide UI elements based on the model's conditioners.
    We do NOT display 'language_id' or 'ctc_loss' even if they exist in the model.
    """
    model = load_model_if_needed(model_choice)
    cond_names = [c.name for c in model.prefix_conditioner.conditioners]
    print("Conditioners in this model:", cond_names)

    text_update = gr.update(visible=("espeak" in cond_names))
    language_update = gr.update(visible=("espeak" in cond_names))
    speaker_audio_update = gr.update(visible=("speaker" in cond_names))
    prefix_audio_update = gr.update(visible=True)
    emotion1_update = gr.update(visible=("emotion" in cond_names))
    emotion2_update = gr.update(visible=("emotion" in cond_names))
    emotion3_update = gr.update(visible=("emotion" in cond_names))
    emotion4_update = gr.update(visible=("emotion" in cond_names))
    emotion5_update = gr.update(visible=("emotion" in cond_names))
    emotion6_update = gr.update(visible=("emotion" in cond_names))
    emotion7_update = gr.update(visible=("emotion" in cond_names))
    emotion8_update = gr.update(visible=("emotion" in cond_names))
    vq_single_slider_update = gr.update(visible=("vqscore_8" in cond_names))
    fmax_slider_update = gr.update(visible=("fmax" in cond_names))
    pitch_std_slider_update = gr.update(visible=("pitch_std" in cond_names))
    speaking_rate_slider_update = gr.update(visible=("speaking_rate" in cond_names))
    dnsmos_slider_update = gr.update(visible=("dnsmos_ovrl" in cond_names))
    speaker_noised_checkbox_update = gr.update(visible=("speaker_noised" in cond_names))
    unconditional_keys_update = gr.update(
        choices=[name for name in cond_names if name not in ("espeak", "language_id")]
    )

    return (
        text_update,
        language_update,
        speaker_audio_update,
        prefix_audio_update,
        emotion1_update,
        emotion2_update,
        emotion3_update,
        emotion4_update,
        emotion5_update,
        emotion6_update,
        emotion7_update,
        emotion8_update,
        vq_single_slider_update,
        fmax_slider_update,
        pitch_std_slider_update,
        speaking_rate_slider_update,
        dnsmos_slider_update,
        speaker_noised_checkbox_update,
        unconditional_keys_update,
    )


def detect_language(text: str) -> str:
    """
    Detect if the text is primarily Japanese, Chinese, or another language.
    Returns: 'ja', 'zh', or 'other'
    """
    # Count characters in different scripts
    jp_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF]', text))  # Hiragana & Katakana
    cn_chars = len(re.findall(r'[\u4E00-\u9FFF]', text))  # Han characters (shared by Japanese & Chinese)
    
    # Check for specific Chinese punctuation that's less common in Japanese
    cn_specific = len(re.findall(r'[，；：""《》【】]', text))
    
    # Check for Japanese-specific particles and endings
    jp_specific = len(re.findall(r'(です|ます|だ|した|ない|ました|でした)', text))
    
    if jp_chars > 0 or jp_specific >= 2:
        return 'ja'
    elif cn_chars > 0 and (cn_specific > 0 or jp_chars == 0):
        return 'zh'
    else:
        return 'other'

def check_tokenizer_available(package_name: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(package_name) is not None

def segment_japanese_text(text: str) -> List[str]:
    """
    Segment Japanese text into sentences using specialized libraries if available.
    Falls back to simpler methods if specialized libraries aren't installed.
    """
    # First choice: spaCy with Japanese model if available
    if check_tokenizer_available('spacy'):
        try:
            import spacy
            try:
                nlp = spacy.load('ja_core_news_sm')
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents]
            except:
                # Japanese model not installed
                pass
        except:
            pass
    
    # Second choice: MeCab if available
    if check_tokenizer_available('fugashi'):
        try:
            import fugashi
            mecab = fugashi.Tagger()
            
            # Custom sentence splitting using MeCab's parsing
            sentences = []
            current_sentence = ""
            
            for word in mecab(text):
                current_sentence += word.surface
                
                # Check for sentence endings (periods, question marks, exclamation marks)
                if word.surface in ["。", "！", "？", "…"] or \
                   (word.pos == "記号" and word.surface in [".", "!", "?"]):
                    sentences.append(current_sentence)
                    current_sentence = ""
            
            # Add the last sentence if there's anything left
            if current_sentence:
                sentences.append(current_sentence)
                
            return sentences
        except:
            pass
            
    # Third choice: Janome if available
    if check_tokenizer_available('janome'):
        try:
            from janome.tokenizer import Tokenizer
            tokenizer = Tokenizer()
            
            # Split on common Japanese sentence endings
            sentences = []
            current_sentence = ""
            
            for token in tokenizer.tokenize(text):
                current_sentence += token.surface
                if token.surface in ["。", "！", "？"] or \
                   (token.part_of_speech.split(',')[0] == "記号" and token.surface in [".", "!", "?"]):
                    sentences.append(current_sentence)
                    current_sentence = ""
            
            # Add the last sentence if there's anything left
            if current_sentence:
                sentences.append(current_sentence)
                
            return sentences
        except:
            pass
    
    # Fallback: Simple rule-based splitting
    sentences = re.split(r'([。！？])', text)
    result = []
    
    # Rejoin the sentences with their punctuation
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
    
    # Add the last part if there's an odd number of elements
    if len(sentences) % 2 == 1 and sentences[-1]:
        result.append(sentences[-1])
        
    return [s for s in result if s.strip()]

def segment_chinese_text(text: str) -> List[str]:
    """
    Segment Chinese text into sentences using specialized libraries if available.
    Falls back to simpler methods if specialized libraries aren't installed.
    """
    # First choice: spaCy with Chinese model if available
    if check_tokenizer_available('spacy'):
        try:
            import spacy
            try:
                nlp = spacy.load('zh_core_web_sm')
                doc = nlp(text)
                return [sent.text.strip() for sent in doc.sents]
            except:
                # Chinese model not installed
                pass
        except:
            pass
    
    # Second choice: Jieba if available
    if check_tokenizer_available('jieba'):
        try:
            import jieba.posseg as pseg
            
            # Custom sentence splitting using punctuation
            sentences = re.split(r'([。！？\!\.。]+)', text)
            result = []
            
            # Rejoin the sentences with their punctuation
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    result.append(sentences[i] + sentences[i+1])
            
            # Add the last part if there's an odd number of elements
            if len(sentences) % 2 == 1 and sentences[-1]:
                result.append(sentences[-1])
                
            return [s for s in result if s.strip()]
        except:
            pass
    
    # Fallback: Simple rule-based splitting
    sentences = re.split(r'([。！？\.!?])', text)
    result = []
    
    # Rejoin the sentences with their punctuation
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            result.append(sentences[i] + sentences[i+1])
    
    # Add the last part if there's an odd number of elements
    if len(sentences) % 2 == 1 and sentences[-1]:
        result.append(sentences[-1])
        
    return [s for s in result if s.strip()]

def get_last_n_words(text: str, n: int = 3) -> str:
    """Extract the last n words from a text."""
    words = text.split()
    return " ".join(words[-n:]) if len(words) >= n else text

def estimate_word_duration_in_tokens(word: str, tokens_per_second: int, speaking_rate: float = 20.0) -> int:
    """
    Estimate how many latent tokens a word might take based on speaking rate.
    """
    logger.debug(f"Estimating duration for word: '{word}' with speaking rate: {speaking_rate}")
    
    # Default rate if none specified
    if speaking_rate <= 0:
        speaking_rate = 20.0
    
    # Cap the rate at the maximum value (40)
    speaking_rate = min(speaking_rate, 40.0)
    
    # Estimate number of phonemes based on character count
    # English averages ~0.7-1.0 phonemes per character
    estimated_phonemes = len(word) * 0.8
    
    # Special case for very short words which often have more phonemes than chars
    if len(word) <= 2:
        estimated_phonemes = max(estimated_phonemes, 2.0)
    
    # Calculate duration in seconds based on speaking rate
    duration_seconds = estimated_phonemes / speaking_rate
    
    # Convert to latent tokens and ensure minimum of 1 token
    tokens = max(1, int(duration_seconds * tokens_per_second))
    
    logger.debug(f"Estimated {tokens} tokens for '{word}'")
    return tokens

def smooth_latent_transition(
    codes_a: torch.Tensor, 
    codes_b: torch.Tensor, 
    overlap_tokens: int,
    smoothing_strength: float = 0.5
) -> torch.Tensor:
    """
    Create a smoother transition between two segments of latent codes
    """
    logger.info(f"Smoothing transition with overlap of {overlap_tokens} tokens, strength {smoothing_strength}")
    start_time = time.time()
    
    # Ensure we don't try to overlap more than we have available
    overlap_tokens = min(overlap_tokens, codes_a.size(-1), codes_b.size(-1))
    logger.debug(f"Using effective overlap of {overlap_tokens} tokens")
    
    # Extract the overlap regions
    overlap_a = codes_a[..., -overlap_tokens:]
    overlap_b = codes_b[..., :overlap_tokens]
    
    # 1. Apply Gaussian smoothing to reduce high-frequency discontinuities
    kernel_size = min(9, overlap_tokens // 2 * 2 + 1)  # Must be odd
    logger.debug(f"Using Gaussian kernel size of {kernel_size}")
    
    if kernel_size >= 3:
        sigma = smoothing_strength * 2.0
        # Reshape for conv1d which expects [N, C, L] format
        orig_shape = overlap_a.shape
        overlap_a_smoothed = overlap_a.float().reshape(-1, 1, overlap_tokens)
        overlap_b_smoothed = overlap_b.float().reshape(-1, 1, overlap_tokens)
        
        # Apply Gaussian smoothing
        overlap_a_smoothed = gaussian_blur1d(overlap_a_smoothed, kernel_size, sigma)
        overlap_b_smoothed = gaussian_blur1d(overlap_b_smoothed, kernel_size, sigma)
        
        # Reshape back
        overlap_a_smoothed = overlap_a_smoothed.reshape(orig_shape)
        overlap_b_smoothed = overlap_b_smoothed.reshape(orig_shape)
    else:
        overlap_a_smoothed = overlap_a.float()
        overlap_b_smoothed = overlap_b.float()
    
    # 2. Find optimal alignment point where similarity is highest
    logger.debug("Finding optimal alignment point")
    similarity_scores = torch.nn.functional.cosine_similarity(
        overlap_a_smoothed.float(), 
        overlap_b_smoothed.float(),
        dim=1
    ).mean(dim=0)
    
    best_pos = torch.argmax(similarity_scores)
    effective_overlap = overlap_tokens - best_pos
    logger.info(f"Found best alignment at position {best_pos}, effective overlap: {effective_overlap}")
    
    # 3. Create smoother crossfade - use S-curve instead of linear blend
    # S-curve fading creates more natural transitions than cosine
    t = torch.linspace(0, 1, effective_overlap, device=codes_a.device)
    fade_curve = 0.5 * (1 + torch.sin(math.pi * (t - 0.5)))
    fade_in = fade_curve.view(1, 1, -1)
    fade_out = 1 - fade_in
    
    # Apply crossfade at the optimal position
    transition_region = (
        overlap_a[..., -effective_overlap:].float() * fade_out + 
        overlap_b[..., best_pos:best_pos+effective_overlap].float() * fade_in
    )
    
    # Construct final output
    non_overlap_a = codes_a[..., :-effective_overlap]
    non_overlap_b = codes_b[..., best_pos+effective_overlap:]
    
    result = torch.cat([
        non_overlap_a, 
        transition_region.to(codes_a.dtype), 
        non_overlap_b
    ], dim=-1)
    
    logger.info(f"Smoothing completed in {time.time() - start_time:.2f}s, output length: {result.shape[-1]}")
    return result

def gaussian_blur1d(x, kernel_size, sigma):
    """Custom implementation of 1D Gaussian blur if torch.nn.functional doesn't have it"""
    channels = x.shape[1]
    
    # Create 1D Gaussian kernel
    kernel_range = torch.arange(kernel_size, device=x.device) - (kernel_size - 1) / 2
    kernel = torch.exp(-0.5 * kernel_range**2 / sigma**2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size).repeat(channels, 1, 1)
    
    # Padding
    padding = (kernel_size - 1) // 2
    padded_input = F.pad(x, (padding, padding), mode='reflect')
    
    # Group convolution
    return F.conv1d(padded_input, kernel, groups=channels)


def generate_with_latent_windows(
    model: Any,
    text: str,
    cond_dict: dict,
    overlap_seconds: float = 0.2,
    cfg_scale: float = 2.0,
    min_p: float = 0.15,
    seed: int = 420,
    prefix_words: int = 3,
    smoothing_strength: float = 0.5
) -> Tuple[int, np.ndarray]:
    """
    Generate audio using sliding windows in latent space with enhanced audio quality
    by reducing popping and noise artifacts at segment boundaries.
    """
    logger.info(f"Starting generation with {len(text)} characters of text")
    logger.info(f"Parameters: overlap={overlap_seconds}s, cfg_scale={cfg_scale}, min_p={min_p}, seed={seed}")
    start_time = time.time()
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer")
        nltk.download('punkt')
    
    # Set global seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Extract speaking rate from conditioning if available
    speaking_rate = 20.0  # Default medium rate
    if 'speaking_rate' in cond_dict:
        speaking_rate = float(cond_dict['speaking_rate'])
    logger.info(f"Using speaking rate: {speaking_rate}")
    
    # Detect language and use appropriate tokenization
    language = detect_language(text) if 'detect_language' in globals() else 'other'
    logger.info(f"Detected language: {language}")
    
    if language == 'ja' and 'segment_japanese_text' in globals():
        sentences = segment_japanese_text(text)
    elif language == 'zh' and 'segment_chinese_text' in globals():
        sentences = segment_chinese_text(text)
    else:
        sentences = nltk.sent_tokenize(text)
    
    # Handle empty or missing sentences
    if not sentences:
        logger.warning("No sentences detected, using full text as a single sentence")
        sentences = [text]
    
    logger.info(f"Split text into {len(sentences)} sentences")
    for i, s in enumerate(sentences):
        logger.debug(f"Sentence {i+1}: {s[:50]}{'...' if len(s) > 50 else ''}")
    
    tokens_per_second = 86
    overlap_size = int(overlap_seconds * tokens_per_second)
    
    if len(sentences) == 1:
        # Single sentence - generate normally
        logger.info("Only one sentence, generating directly")
        torch.manual_seed(seed)
        conditioning = model.prepare_conditioning(cond_dict)
        codes = model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=86 * 120,  # 120 seconds max
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=dict(min_p=min_p)
        )
        logger.info(f"Generated {codes.shape[-1]} tokens ({codes.shape[-1]/tokens_per_second:.2f}s of audio)")
        wav_out = model.autoencoder.decode(codes).cpu().detach()
        logger.info(f"Generation completed in {time.time() - start_time:.2f}s")
        return model.autoencoder.sampling_rate, wav_out.squeeze().numpy()

    # For multiple sentences, we need to process each one
    all_codes = []
    last_generated_codes = None
    
    for i, sentence in enumerate(sentences):
        logger.info(f"Processing sentence {i+1}/{len(sentences)}: {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
        sentence_start_time = time.time()
        
        # Create conditioning for this sentence
        sent_dict = cond_dict.copy()
        
        if i > 0:
            # Extract last few words from previous sentence as prefix text
            prefix_text = get_last_n_words(sentences[i-1], prefix_words)
            logger.info(f"Using prefix from previous sentence: '{prefix_text}'")
            
            # Estimate tokens needed for prefix using speaking rate
            prefix_tokens = estimate_word_duration_in_tokens(
                prefix_text, tokens_per_second, speaking_rate
            )
            
            # Ensure minimum overlap
            prefix_tokens = max(prefix_tokens, overlap_size)
            logger.debug(f"Using {prefix_tokens} tokens for prefix")
            
            # Take audio codes from the end of previous generation as prefix
            audio_prefix_codes = last_generated_codes[..., -prefix_tokens:]
            
            # Create prefixed text for TTS
            prefixed_sentence = f"{prefix_text} {sentence}"
            sent_dict['espeak'] = ([prefixed_sentence], [cond_dict['espeak'][1][0]])
            
            # Generate with audio prefix
            logger.info(f"Generating with audio prefix ({prefix_tokens} tokens)")
            torch.manual_seed(seed)
            conditioning = model.prepare_conditioning(sent_dict)
            max_tokens = int(len(sentence) * 1.5 * tokens_per_second / (10 * (speaking_rate/20)))
            logger.debug(f"max_new_tokens = {max_tokens}")
            
            codes = model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=audio_prefix_codes,
                max_new_tokens=max_tokens,
                cfg_scale=cfg_scale,
                batch_size=1,
                sampling_params=dict(min_p=min_p)
            )
            
            logger.info(f"Generated {codes.shape[-1]} tokens for sentence {i+1}")
            
            # For the second segment onwards, we add without the prefix portion
            new_segment = codes[..., prefix_tokens:]
            all_codes.append(new_segment)
            
            # Store full segment for next iteration's prefix
            last_generated_codes = codes
            
        else:
            # First sentence - generate normally
            sent_dict['espeak'] = ([sentence], [cond_dict['espeak'][1][0]])
            logger.info("Generating first sentence")
            torch.manual_seed(seed)
            conditioning = model.prepare_conditioning(sent_dict)
            max_tokens = int(len(sentence) * 1.5 * tokens_per_second / (10 * (speaking_rate/20)))
            logger.debug(f"max_new_tokens = {max_tokens}")
            
            codes = model.generate(
                prefix_conditioning=conditioning,
                max_new_tokens=max_tokens,
                cfg_scale=cfg_scale,
                batch_size=1,
                sampling_params=dict(min_p=min_p)
            )
            
            logger.info(f"Generated {codes.shape[-1]} tokens for sentence {i+1}")
            all_codes.append(codes)
            last_generated_codes = codes
        
        logger.info(f"Sentence {i+1} processed in {time.time() - sentence_start_time:.2f}s")
        
        # Check total length
        total_length = sum(c.shape[-1] for c in all_codes)
        logger.info(f"Current total: {total_length} tokens ({total_length/tokens_per_second:.2f}s)")
        
        if total_length >= tokens_per_second * 120:  # 120 seconds max
            logger.warning("Reached maximum length, truncating remaining sentences")
            break
    
    # Concatenate all segments
    logger.info(f"Concatenating {len(all_codes)} segments")
    final_codes = torch.cat(all_codes, dim=-1)
    final_codes = final_codes.to(torch.long)
    
    # Decode to audio
    logger.info(f"Decoding {final_codes.shape[-1]} tokens to audio")
    wav_out = model.autoencoder.decode(final_codes).cpu().detach()
    
    logger.info(f"Total generation completed in {time.time() - start_time:.2f}s")
    return model.autoencoder.sampling_rate, wav_out.squeeze().numpy()


def generate_audio(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1, e2, e3, e4, e5, e6, e7, e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    top_p,
    top_k,
    min_p,
    linear,
    confidence,
    quadratic,
    seed,
    randomize_seed,
    unconditional_keys,
    use_windowing,
    progress=gr.Progress(),
):
    """
    Non-streaming generation: generate codes, decode, and return the full audio.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    top_p = float(top_p)
    top_k = int(top_k)
    min_p = float(min_p)
    linear = float(linear)
    confidence = float(confidence)
    quadratic = float(quadratic)
    seed = int(seed)
    max_new_tokens = 86 * 60

    # This is a bit ew, but works for now.
    global SPEAKER_AUDIO_PATH, SPEAKER_EMBEDDING

    if randomize_seed:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(seed)

    if speaker_audio is not None and "speaker" not in unconditional_keys:
        if speaker_audio != SPEAKER_AUDIO_PATH:
            print("Recomputed speaker embedding")
            wav, sr = torchaudio.load(speaker_audio)
            SPEAKER_EMBEDDING = selected_model.make_speaker_embedding(wav, sr)
            SPEAKER_EMBEDDING = SPEAKER_EMBEDDING.to(device, dtype=torch.bfloat16)
            SPEAKER_AUDIO_PATH = speaker_audio

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)

    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)
    
    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=SPEAKER_EMBEDDING,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    estimated_generation_duration = 30 * len(text) / 400
    estimated_total_steps = int(estimated_generation_duration * 86)

    def update_progress(_frame: torch.Tensor, step: int, _total_steps: int) -> bool:
        progress((step, estimated_total_steps))
        return True

    if use_windowing:
        return generate_with_latent_windows(
            selected_model,
            text,
            cond_dict,
            overlap_seconds=0.1,
            cfg_scale=cfg_scale,
            min_p=min_p,
            seed=seed
        ), seed
    else:
        codes = selected_model.generate(
            prefix_conditioning=conditioning,
            audio_prefix_codes=audio_prefix_codes,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
            callback=update_progress,
        )
        wav_out = selected_model.autoencoder.decode(codes).cpu().detach()
        sr_out = selected_model.autoencoder.sampling_rate
        if wav_out.dim() == 2 and wav_out.size(0) > 1:
            wav_out = wav_out[0:1, :]
        return (sr_out, wav_out.squeeze().numpy()), seed
    
def numpy_to_mp3(audio_array, sampling_rate):
    # Normalize audio_array if it's floating-point
    if np.issubdtype(audio_array.dtype, np.floating):
        max_val = np.max(np.abs(audio_array))
        audio_array = (audio_array / max_val) * 32767 # Normalize to 16-bit range
        audio_array = audio_array.astype(np.int16)

    # Create an audio segment from the numpy array
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )

    # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
    mp3_io = io.BytesIO()
    audio_segment.export(mp3_io, format="mp3", bitrate="320k")

    # Get the MP3 bytes
    mp3_bytes = mp3_io.getvalue()
    mp3_io.close()

    return mp3_bytes

def generate_audio_stream(
    model_choice,
    text,
    language,
    speaker_audio,
    prefix_audio,
    e1,
    e2,
    e3,
    e4,
    e5,
    e6,
    e7,
    e8,
    vq_single,
    fmax,
    pitch_std,
    speaking_rate,
    dnsmos_ovrl,
    speaker_noised,
    cfg_scale,
    min_p,
    seed,
    randomize_seed,
    unconditional_keys,
    chunk_size,
):
    """
    Streaming generation: a generator function that yields (sampling_rate, audio_chunk)
    tuples. The stream is constructed by incrementally decoding the latent codes.
    Each yielded chunk is converted to int16.
    """
    selected_model = load_model_if_needed(model_choice)

    speaker_noised_bool = bool(speaker_noised)
    fmax = float(fmax)
    pitch_std = float(pitch_std)
    speaking_rate = float(speaking_rate)
    dnsmos_ovrl = float(dnsmos_ovrl)
    cfg_scale = float(cfg_scale)
    min_p = float(min_p)
    seed = int(seed)
    max_new_tokens = 86 * 30
    chunk_size = 40

    if randomize_seed:
        seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
    torch.manual_seed(seed)

    speaker_embedding = None
    if speaker_audio is not None and "speaker" not in unconditional_keys:
        wav, sr = torchaudio.load(speaker_audio)
        speaker_embedding = selected_model.make_speaker_embedding(wav, sr)
        speaker_embedding = speaker_embedding.to(device, dtype=torch.bfloat16)

    audio_prefix_codes = None
    if prefix_audio is not None:
        wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        wav_prefix = torchaudio.functional.resample(
            wav_prefix, sr_prefix, selected_model.autoencoder.sampling_rate
        )
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        with torch.autocast(device, dtype=torch.float32):
            audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))

    emotion_tensor = torch.tensor(list(map(float, [e1, e2, e3, e4, e5, e6, e7, e8])), device=device)
    vq_val = float(vq_single)
    vq_tensor = torch.tensor([vq_val] * 8, device=device).unsqueeze(0)

    cond_dict = make_cond_dict(
        text=text,
        language=language,
        speaker=speaker_embedding,
        emotion=emotion_tensor,
        vqscore_8=vq_tensor,
        fmax=fmax,
        pitch_std=pitch_std,
        speaking_rate=speaking_rate,
        dnsmos_ovrl=dnsmos_ovrl,
        speaker_noised=speaker_noised_bool,
        device=device,
        unconditional_keys=unconditional_keys,
    )
    conditioning = selected_model.prepare_conditioning(cond_dict)

    # Iterate over the model's generate_stream() output.
    for sr_out, audio_chunk in selected_model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=audio_prefix_codes,
        max_new_tokens=max_new_tokens,
        cfg_scale=cfg_scale,
        batch_size=1,
        sampling_params=dict(min_p=min_p),
        chunk_size=chunk_size,
    ):
        # audio_chunk is expected to be a numpy array in float32.
        
        yield numpy_to_mp3(audio_chunk, sampling_rate=sr_out), seed


def build_interface():
    supported_models = []
    if "transformer" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-transformer")

    if "hybrid" in ZonosBackbone.supported_architectures:
        supported_models.append("Zyphra/Zonos-v0.1-hybrid")
    else:
        print(
            "| The current ZonosBackbone does not support the hybrid architecture, meaning only the transformer model will be available in the model selector.\n"
            "| This probably means the mamba-ssm library has not been installed."
        )

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                model_choice = gr.Dropdown(
                    choices=supported_models,
                    value=supported_models[0],
                    label="Zonos Model Type",
                    info="Select the model variant to use.",
                )
                text = gr.Textbox(
                    label="Text to Synthesize",
                    value="Zonos uses eSpeak for text to phoneme conversion!",
                    lines=4,
                    max_length=500,
                )
                language = gr.Dropdown(
                    choices=supported_language_codes,
                    value="en-us",
                    label="Language Code",
                    info="Select a language code.",
                )
            prefix_audio = gr.Audio(
                value="assets/silence_100ms.wav",
                label="Optional Prefix Audio (continue from this audio)",
                type="filepath",
            )
            with gr.Column():
                speaker_audio = gr.Audio(
                    label="Optional Speaker Audio (for cloning)",
                    type="filepath",
                )
                speaker_noised_checkbox = gr.Checkbox(label="Denoise Speaker?", value=False)

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Conditioning Parameters")
                dnsmos_slider = gr.Slider(1.0, 5.0, value=4.0, step=0.1, label="DNSMOS Overall")
                fmax_slider = gr.Slider(0, 24000, value=24000, step=1, label="Fmax (Hz)")
                vq_single_slider = gr.Slider(0.5, 0.8, 0.78, 0.01, label="VQ Score")
                pitch_std_slider = gr.Slider(0.0, 300.0, value=45.0, step=1, label="Pitch Std")
                speaking_rate_slider = gr.Slider(5.0, 30.0, value=15.0, step=0.5, label="Speaking Rate")
            with gr.Column():
                gr.Markdown("## Generation Parameters")
                cfg_scale_slider = gr.Slider(1.0, 5.0, 2.0, 0.1, label="CFG Scale")
                seed_number = gr.Number(label="Seed", value=420, precision=0)
                randomize_seed_toggle = gr.Checkbox(label="Randomize Seed (before generation)", value=True)
                chunk_size = gr.Slider(10, 100, value=40, step=5, label="Chunk Size")

        with gr.Accordion("Sampling", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### NovelAi's unified sampler")
                    linear_slider = gr.Slider(-2.0, 2.0, 0.5, 0.01, label="Linear (set to 0 to disable unified sampling)", info="High values make the output less random.")
                    #Conf's theoretical range is between -2 * Quad and 0.
                    confidence_slider = gr.Slider(-2.0, 2.0, 0.40, 0.01, label="Confidence", info="Low values make random outputs more random.")
                    quadratic_slider = gr.Slider(-2.0, 2.0, 0.00, 0.01, label="Quadratic", info="High values make low probablities much lower.")
                with gr.Column():
                    gr.Markdown("### Legacy sampling")
                    top_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Top P")
                    min_k_slider = gr.Slider(0.0, 1024, 0, 1, label="Min K")
                    min_p_slider = gr.Slider(0.0, 1.0, 0, 0.01, label="Min P")

        with gr.Accordion("Advanced Parameters", open=False):
            gr.Markdown(
                "### Unconditional Toggles\n"
                "Checking a box will make the model ignore the corresponding conditioning value and make it unconditional.\n"
                'Practically this means the given conditioning feature will be unconstrained and "filled in automatically".'
            )
            with gr.Row():
                unconditional_keys = gr.CheckboxGroup(
                    [
                        "speaker",
                        "emotion",
                        "vqscore_8",
                        "fmax",
                        "pitch_std",
                        "speaking_rate",
                        "dnsmos_ovrl",
                        "speaker_noised",
                    ],
                    value=["emotion"],
                    label="Unconditional Keys",
                )
            gr.Markdown(
                "### Latent Windowing\n"
                "This feature processes longer texts by breaking them into sentences and generating them separately with smooth transitions in latent space. "
                "It works best when used with voice cloning (speaker audio) using Transformers (not hybrid) and may produce inconsistent results otherwise. "
                "Enable this if you're doing voice cloning with longer texts."
            )
            use_windowing = gr.Checkbox(label="Enable Latent Windowing", value=False)

            gr.Markdown(
                "### Emotion Sliders\n"
                "Warning: The way these sliders work is not intuitive and may require some trial and error to get the desired effect.\n"
                "Certain configurations can cause the model to become unstable. Setting emotion to unconditional may help."
            )
            with gr.Row():
                emotion1 = gr.Slider(0.0, 1.0, 1.0, 0.05, label="Happiness")
                emotion2 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Sadness")
                emotion3 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Disgust")
                emotion4 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Fear")
            with gr.Row():
                emotion5 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Surprise")
                emotion6 = gr.Slider(0.0, 1.0, 0.05, 0.05, label="Anger")
                emotion7 = gr.Slider(0.0, 1.0, 0.1, 0.05, label="Other")
                emotion8 = gr.Slider(0.0, 1.0, 0.2, 0.05, label="Neutral")

        with gr.Column():
            generate_button = gr.Button("Generate Audio")
            output_audio = gr.Audio(label="Generated Audio", type="numpy", autoplay=True)

        with gr.Tab("Stream Audio"):
            stream_button = gr.Button("Stream Audio", variant="primary")
            stream_output = gr.Audio(label="Streaming Audio", type="numpy", streaming=True, autoplay=True)

        model_choice.change(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # On page load, trigger the same UI refresh.
        demo.load(
            fn=update_ui,
            inputs=[model_choice],
            outputs=[
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                unconditional_keys,
            ],
        )

        # Generate audio on button click.
        generate_button.click(
            fn=generate_audio,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                top_p_slider,
                min_k_slider,
                min_p_slider,
                linear_slider,
                confidence_slider,
                quadratic_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
                use_windowing,
            ],
            outputs=[output_audio, seed_number],
        )

        # Stream audio on stream button click.
        stream_button.click(
            fn=generate_audio_stream,
            inputs=[
                model_choice,
                text,
                language,
                speaker_audio,
                prefix_audio,
                emotion1,
                emotion2,
                emotion3,
                emotion4,
                emotion5,
                emotion6,
                emotion7,
                emotion8,
                vq_single_slider,
                fmax_slider,
                pitch_std_slider,
                speaking_rate_slider,
                dnsmos_slider,
                speaker_noised_checkbox,
                cfg_scale_slider,
                min_p_slider,
                seed_number,
                randomize_seed_toggle,
                unconditional_keys,
                chunk_size,
            ],
            outputs=[stream_output, seed_number],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    share = getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
