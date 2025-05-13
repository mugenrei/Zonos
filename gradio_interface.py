import os
import re
import tempfile
import scipy.io.wavfile
import torch
import torchaudio
import torchaudio.functional as F
import gradio as gr
import numpy as np
from os import getenv
import io
from pydub import AudioSegment
import numpy as np
import nltk
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import importlib.util
from typing import Tuple, List
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
    level=logging.INFO,
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
    
class WhisperASR:
    """Whisper ASR with word-level timestamp support"""
    def __init__(self, model_id="openai/whisper-small", device=None):
        self.model_id = model_id
        self.device = device if device is not None else "cuda" if torch.cuda.is_available() else "cpu"
        self.initialized = False
        self.model = None
        self.processor = None
        self.asr_pipeline = None
        # Whisper models require 16kHz sampling rate
        self.target_sr = 16000
        logger.info(f"Initialized WhisperASR instance with model_id={model_id}, device={self.device}")
        
    def initialize(self):
        """Lazy initialization of the ASR model"""
        if not self.initialized:
            try:
                logger.info(f"Initializing WhisperASR model {self.model_id} on {self.device}")
                start_time = time.time()
                
                # Check if transformers is available
                import transformers
                
                # Suppress warnings from Transformers
                transformers.logging.set_verbosity_error()
                
                # Initialize the model and processor
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id, 
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="eager"
                )
                self.model.to(self.device)
                
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                
                self.asr_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    max_new_tokens=128,
                    chunk_length_s=30,
                    batch_size=8,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device=self.device,
                )
                self.initialized = True
                logger.info(f"WhisperASR model initialized successfully in {time.time() - start_time:.2f}s")
                return True
            except (ImportError, Exception) as e:
                logger.error(f"Failed to initialize ASR: {str(e)}")
                return False
        return True
    
    def _resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio to the target sampling rate (16kHz for Whisper)"""
        if orig_sr == self.target_sr:
            return audio
            
        logger.debug(f"Resampling audio from {orig_sr}Hz to {self.target_sr}Hz")
        
        # Convert numpy array to tensor
        audio_tensor = torch.tensor(audio).float()
        
        # Resample using torchaudio
        resampled = F.resample(
            audio_tensor,
            orig_freq=orig_sr,
            new_freq=self.target_sr
        ).numpy()
        
        logger.debug(f"Resampled audio from {len(audio)} to {len(resampled)} samples")
        return resampled
        
def process_prefix_audio(prefix_audio, selected_model, device):
    """
    Process prefix audio by saving to temp file and reloading to ensure proper format
    
    Args:
        prefix_audio: The numpy array containing audio data
        selected_model: The model containing the autoencoder
        device: The torch device to use
    
    Returns:
        Encoded audio codes
    """
    # Create a temporary directory that will be automatically cleaned up
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a temporary file path
        temp_file = os.path.join(temp_dir, "temp_audio.wav")
        
        # Save the numpy array as a WAV file
        # First ensure it's the right dtype for saving
        if prefix_audio.dtype != np.float32:
            prefix_audio = prefix_audio.astype(np.float32)
        
        # Save as WAV file - assume 44100Hz if not specified otherwise
        temp_sr = 44100  # Default sample rate
        torchaudio.save(temp_file, 
                        torch.from_numpy(prefix_audio).unsqueeze(0), 
                        temp_sr)
        
        # Load it back using torchaudio
        wav_prefix, sr_prefix = torchaudio.load(temp_file)
        
        # Average if stereo to mono
        wav_prefix = wav_prefix.mean(0, keepdim=True)
        
        # Preprocess with the model's autoencoder
        wav_prefix = selected_model.autoencoder.preprocess(wav_prefix, sr_prefix)
        wav_prefix = wav_prefix.to(device, dtype=torch.float32)
        
        # Encode and return
        audio_prefix_codes = selected_model.autoencoder.encode(wav_prefix.unsqueeze(0))
        
        return audio_prefix_codes

def trim_prefix_from_generated_audio(generated_audio, prefix_audio, asr_model, sampling_rate, prefix_text=None):
    """
    Identifies and removes the prefix portion from the beginning of generated audio.
    
    Args:
        generated_audio: The generated audio containing the prefix
        prefix_audio: The audio used as prefix
        asr_model: The ASR model used for word timestamps
        sampling_rate: The audio sampling rate
        prefix_text: Optional text of the prefix for verification
        
    Returns:
        Trimmed audio without the prefix
    """
    logger.info("Trimming prefix from generated audio")
    
    # Method 1: Fixed length approach - simply remove the prefix length
    prefix_samples = len(prefix_audio)
    if prefix_samples >= len(generated_audio):
        logger.warning("Prefix is longer than generated audio, using 50% of generated audio")
        prefix_samples = len(generated_audio) // 2
    
    # Method 2: Use ASR to detect where the prefix ends and new content begins
    try:
        # Get word timestamps for the generated audio
        word_timestamps = get_word_timestamps(asr_model, generated_audio, sampling_rate, prefix_text)
        
        if prefix_text and word_timestamps and len(word_timestamps) > 1:
            prefix_words = prefix_text.strip().split()
            num_prefix_words = len(prefix_words)
            
            # Look for the end timestamp of the last prefix word
            if len(word_timestamps) > num_prefix_words:
                # Use the end of the last prefix word as the cut point
                cut_time = word_timestamps[num_prefix_words-1]['end']
                cut_sample = int(cut_time * sampling_rate)
                
                # Sanity check - ensure we're not cutting too much or too little
                if cut_sample > 0 and cut_sample < len(generated_audio) * 0.5:
                    prefix_samples = cut_sample
                    logger.info(f"Using ASR-based prefix trimming: {cut_time:.2f}s ({cut_sample} samples)")
                else:
                    logger.warning(f"ASR cut point at {cut_time:.2f}s seems unreliable, using direct prefix length")
    except Exception as e:
        logger.warning(f"ASR-based trimming failed: {str(e)}, using direct prefix length")
    
    # Apply a small crossfade to avoid clicks at the transition
    crossfade_samples = min(int(0.05 * sampling_rate), prefix_samples // 2)  # 50ms crossfade or less
    
    if crossfade_samples > 0 and prefix_samples + crossfade_samples < len(generated_audio):
        # Create a linear fade-in window
        fade_in = np.linspace(0, 1, crossfade_samples)
        fade_out = np.linspace(1, 0, crossfade_samples)
        
        # Apply crossfade
        generated_audio[prefix_samples:prefix_samples+crossfade_samples] *= fade_in
        generated_audio[prefix_samples-crossfade_samples:prefix_samples] *= fade_out
    
    # Return the trimmed audio
    trimmed_audio = generated_audio[prefix_samples:]
    logger.info(f"Trimmed {prefix_samples/sampling_rate:.2f}s from beginning of generated audio")
    return trimmed_audio

def generate_with_sentence_prefixing(
    model,
    text: str,
    cond_dict: dict,
    cfg_scale: float = 2.0,
    seed: int = 420,
    language: str = None,
    batch_size: int = 1,
    sampling_params: dict = None,
    callback = None,
    prefix_words: int = 3,
    device: str = None,
    log_level: int = logging.INFO,
    whisper_model_id: str = "openai/whisper-large-v3"
) -> Tuple[int, np.ndarray]:
    """
    Generate audio by processing sentences with word timestamping ASR-based prefix continuity
    
    Args:
        model: The Zonos model
        text: Input text to synthesize
        cond_dict: Conditioning dictionary
        cfg_scale: Classifier-free guidance scale
        min_p: Min p parameter for generation
        seed: Random seed
        language: Language code from cond_dict
        prefix_words: Number of words to use as prefix (1-3 recommended)
        device: Device to run on
        log_level: Logging level (default: INFO)
        whisper_model_id: ID of the Whisper model to use
        
    Returns:
        Tuple of (sample_rate, audio_waveform)
    """
    lang_dict = {'en-us': 'en', 'en-gb': 'en', 'ja': 'ja', 'zh': 'zh'}
    # Set logger level for this run
    logger.setLevel(log_level)
    
    # Track overall execution time
    overall_start_time = time.time()
    
    logger.info(f"Starting audio generation for text ({len(text)} chars) with sentence prefixing")
    logger.debug(f"Parameters: cfg_scale={cfg_scale}, seed={seed}, prefix_words={prefix_words}")
    
    # Set device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.debug(f"Using device: {device}")
    
    # Set global seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Map language code to standard code
    asr_language = lang_dict.get(language, 'en') if language else 'en'
    logger.info(f"Using language: {asr_language} for processing")
    
    tokenization_start = time.time()
    if asr_language == 'ja':
        sentences = segment_japanese_text(text)
    elif asr_language == 'zh':
        sentences = segment_chinese_text(text)
    else:
        try:
            nltk.data.find('tokenizers/punkt')
            logger.debug("NLTK punkt tokenizer found")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer")
            nltk.download('punkt')
        sentences = nltk.sent_tokenize(text)
    
    # Handle empty or missing sentences
    if not sentences:
        logger.warning("No sentences detected, using full text as a single sentence")
        sentences = [text]
    
    logger.info(f"Text split into {len(sentences)} sentences in {time.time() - tokenization_start:.2f}s")
    
    # Initialize ASR
    asr_init_start = time.time()
    asr_model = WhisperASR(model_id=whisper_model_id, device=device)
    use_asr_prefix = asr_model.initialize()
    
    if use_asr_prefix:
        logger.info(f"ASR model initialized in {time.time() - asr_init_start:.2f}s")
    else:
        logger.warning("Failed to initialize ASR model")
        return None, None
    
    # List to store generated audio segments
    all_audio_segments = []
    sampling_rate = model.autoencoder.sampling_rate
    previous_audio = None
    previous_sentence = None
    
    # Generate audio for each sentence
    for i, sentence in enumerate(sentences):
        current_sentence = sentence
        
        logger.info(f"Processing sentence {i+1}/{len(sentences)}: {current_sentence[:50]}{'...' if len(current_sentence) > 50 else ''}")
        
        # Handle prefixing for all but the first sentence
        prefix_audio = None
        prefix_text = None
        if i > 0 and previous_audio is not None and previous_sentence is not None and use_asr_prefix:
            # Extract the last few words from the previous sentence
            last_few_words = extract_last_n_words(previous_sentence, prefix_words)
            if last_few_words:
                logger.debug(f"Extracted prefix words: '{last_few_words}'")
                
                # Get timestamps for these words in the previous audio
                word_timestamps = get_word_timestamps(asr_model, previous_audio, sampling_rate, last_few_words)
                
                if word_timestamps:
                    # Extract the audio snippet for these words
                    start_time = word_timestamps[0]['start']
                    end_time = word_timestamps[-1]['end']
                    
                    # Convert time to samples
                    start_sample = int(start_time * sampling_rate)
                    end_sample = int(end_time * sampling_rate)
                    
                    # Extract the audio segment
                    prefix_audio = previous_audio[start_sample:end_sample]
                    prefix_text = last_few_words
                    
                    logger.info(f"Created audio prefix from previous sentence: '{prefix_text}' ({len(prefix_audio)/sampling_rate:.2f}s)")
        
        # Generate audio for the current sentence
        generation_start = time.time()
        torch.manual_seed(seed)
        
        # Prepare conditioning
        sentence_dict = cond_dict.copy()
        sentence_dict['espeak'] = ([current_sentence], [cond_dict['espeak'][1][0]])
        conditioning = model.prepare_conditioning(sentence_dict)
        
        # Generate with or without prefix
        if prefix_audio is not None and prefix_text is not None:
            prefix_codes = process_prefix_audio(prefix_audio, model, device)
            
            # Generate with prefix
            codes = model.generate(
                prefix_conditioning=conditioning,
                audio_prefix_codes=prefix_codes,
                max_new_tokens=86 * 30,  # 30 seconds max
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                sampling_params=sampling_params,
                callback=callback,
            )
        else:
            # Generate without prefix
            codes = model.generate(
                prefix_conditioning=conditioning,
                max_new_tokens=86 * 30,  # 30 seconds max
                cfg_scale=cfg_scale,
                batch_size=batch_size,
                sampling_params=sampling_params,
                callback=callback,
            )
        
        # Decode the generated audio
        wav_out = model.autoencoder.decode(codes).cpu().detach().numpy()[0]

        # Check the shape of the output audio
        logger.debug(f"Generated audio shape: {wav_out.shape}")

        # Ensure wav_out is 1D, flatten if necessary
        if len(wav_out.shape) > 1:
            original_shape = wav_out.shape
            wav_out = wav_out.flatten()
            logger.info(f"Flattened audio from shape {original_shape} to {wav_out.shape}")

        # Trim prefix from the generated audio if prefix was used
        if prefix_audio is not None and prefix_text is not None:
            wav_out = trim_prefix_from_generated_audio(
                generated_audio=wav_out, 
                prefix_audio=prefix_audio,
                asr_model=asr_model, 
                sampling_rate=sampling_rate,
                prefix_text=prefix_text
            )

        logger.info(f"Generation for sentence {i+1} completed in {time.time() - generation_start:.2f}s")

        # Store this audio segment
        all_audio_segments.append(wav_out)
        
        # Keep track of this sentence and its audio for the next iteration
        previous_audio = wav_out
        previous_sentence = current_sentence
    
    # Join all audio segments
    logger.info("Joining audio segments...")
    try:
        # First make sure all segments are 1D arrays
        for i, segment in enumerate(all_audio_segments):
            if len(segment.shape) > 1:
                all_audio_segments[i] = segment.flatten()
                logger.debug(f"Flattened segment {i} from shape {segment.shape} to {all_audio_segments[i].shape}")
            
        final_audio = np.concatenate(all_audio_segments)
        logger.info(f"Successfully joined {len(all_audio_segments)} audio segments")
    except ValueError as e:
        logger.error(f"Error joining audio segments: {e}")
        return None, None
    
    # Save audio
    output_filename = "/mnt/g/AI/Zonos-fork/generated_audio.wav"
    logger.debug(f"Saving audio to {output_filename}")
    try:
        scipy.io.wavfile.write(output_filename, sampling_rate, final_audio.astype(np.float32))
        logger.info(f"Audio successfully saved to {output_filename}")
    except Exception as e:
        logger.error(f"Failed to save audio: {str(e)}")
    
    # Log final stats
    audio_duration = len(final_audio) / sampling_rate
    logger.info(f"Total audio length: {audio_duration:.2f}s")
    logger.info(f"Overall processing completed in {time.time() - overall_start_time:.2f}s")
    
    return sampling_rate, final_audio

# Helper function to extract the last N words from a sentence
def extract_last_n_words(text, n=3):
    """Extract the last n words from the given text."""
    words = text.split()
    if len(words) <= n:
        return text
    return ' '.join(words[-n:])

# Helper function to get word timestamps using the ASR model
def get_word_timestamps(asr_model, audio, sample_rate, target_words):
    """
    Use ASR to get timestamps for specific words in the audio
    
    Args:
        asr_model: Initialized WhisperASR model
        audio: Audio array
        sample_rate: Sample rate of the audio
        target_words: The words to find timestamps for (as a string) - used as a hint
                      for how many words to return from the end of the transcription
    
    Returns:
        List of word timestamps with start and end times for the last N words
    """
    logger.debug(f"Getting word timestamps for target words: '{target_words}'")
    logger.debug(f"Audio shape: {audio.shape}, Sample rate: {sample_rate}")
    
    # Make sure the ASR model is initialized
    if not asr_model.initialized:
        logger.debug("ASR model not initialized, initializing now")
        if not asr_model.initialize():
            logger.error("Failed to initialize ASR model for word timestamps")
            return []
    
    # Resample audio if needed
    logger.debug(f"Resampling audio from {sample_rate}Hz to {asr_model.target_sr}Hz")
    audio_resampled = asr_model._resample_audio(audio, sample_rate)
    logger.debug(f"Resampled audio from {len(audio)} to {len(audio_resampled)} samples")
    logger.debug(f"Resampled audio shape: {audio_resampled.shape}")
    
    # Run ASR with word timestamps
    try:
        logger.debug("Running ASR pipeline with word timestamps")
        result = asr_model.asr_pipeline(
            audio_resampled,
            return_timestamps="word",
            chunk_length_s=30,
            stride_length_s=5
        )
        logger.debug(f"ASR pipeline returned {len(result)} results")
        logger.debug(f"ASR result: {result}")
        
        # Extract word timestamps
        word_timestamps = []
        
        # Estimate how many words we're looking for based on the target_words
        target_word_count = len(target_words.split())
        
        # Check if the response has 'chunks' field (based on the log format)
        if 'chunks' in result:
            chunks = result['chunks']
            logger.debug(f"ASR returned {len(chunks)} chunks with timestamps")
            
            # Get the last N chunks, where N is approximately the number of words in target_words
            last_n_chunks = chunks[-target_word_count:] if len(chunks) > target_word_count else chunks
            
            # Process each of the last N chunks
            for chunk in last_n_chunks:
                text = chunk['text'].strip()
                timestamp = chunk['timestamp']
                
                if timestamp[1] is None:
                    # If end timestamp is None, estimate it
                    # For the last chunk, this is especially important
                    if chunk == chunks[-1]:
                        # If this is the last chunk, estimate a reasonable duration
                        end_time = timestamp[0] + 0.5  # Default duration if none provided
                    else:
                        # If not the last chunk, use the next chunk's start time
                        next_idx = chunks.index(chunk) + 1
                        if next_idx < len(chunks):
                            end_time = chunks[next_idx]['timestamp'][0]
                        else:
                            end_time = timestamp[0] + 0.5
                else:
                    end_time = timestamp[1]
                
                logger.debug(f"Including chunk: '{text}' (start: {timestamp[0]:.2f}s, end: {end_time:.2f}s)")
                word_timestamps.append({
                    "word": text,
                    "start": timestamp[0],
                    "end": end_time
                })
            
            # If we found timestamps for words, return them
            if len(word_timestamps) > 0:
                logger.debug(f"Found {len(word_timestamps)} chunks at the end of the transcription")
                return word_timestamps
        else:
            logger.debug("No 'chunks' field in ASR result")
            return []
    
    except Exception as e:
        logger.error(f"Error getting word timestamps: {str(e)}")
        logger.debug(f"Exception details: {repr(e)}")
    
    # Fallback if we couldn't find timestamps
    logger.warning(f"Could not find timestamps for ending words similar to: '{target_words}'")
    return []

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
    use_sentencing,
    whisper_model_choice,
    prefix_words,
    log_level,
    progress=gr.Progress(),
):
    """
    Non-streaming generation: generate codes, decode, and return the full audio.
    """
    selected_model = load_model_if_needed(model_choice)
    log_level_dict = {"Debug": logging.DEBUG, "Info": logging.INFO, "Warning": logging.WARNING, "Error": logging.ERROR}
    sentencing_dict = {
        "whisper_model_choice": whisper_model_choice,
        "prefix_words": prefix_words,
        "log_level": log_level_dict[log_level]
    }
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

    if use_sentencing:
        return generate_with_sentence_prefixing(
            selected_model,
            text,
            cond_dict,
            cfg_scale=cfg_scale,
            seed=seed,
            language=language,
            batch_size=1,
            sampling_params=dict(top_p=top_p, top_k=top_k, min_p=min_p, linear=linear, conf=confidence, quad=quadratic),
            callback=update_progress,
            prefix_words=sentencing_dict.get("prefix_words", 3),
            device=device,
            log_level=sentencing_dict.get("log_level", logging.INFO),
            whisper_model_id=sentencing_dict.get("whisper_model_choice", "openai/whisper-large-v3")
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
                "### Sentence Prefixing\n"
                "This feature uses Whisper to detect the end of the prefix audio and the start of the new content.\n"
                "It can be useful for generating long audio sequences with a smooth transition between segments."
            )
            use_sentencing = gr.Checkbox(label="Enable Sentence Prefixing", value=False)
            whisper_models = [
                "openai/whisper-tiny",
                "openai/whisper-small",
                "openai/whisper-medium",
                "openai/whisper-large-v3",
                "openai/whisper-large-v3-turbo",
            ]
            log_level_list = ["Debug", "Info", "Warning", "Error"]
            with gr.Row():
                with gr.Column():
                    whisper_model_choice = gr.Dropdown(
                        choices=whisper_models,
                        value=whisper_models[3],
                        label="Whisper Model",
                        info="Select the model to use.",
                    )
                with gr.Column():
                    prefix_words = gr.Number(label="Number of prefix words", value=3, precision=0)
                    log_level = gr.Dropdown(choices=log_level_list, value="Info", label="Log Level")
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
                use_sentencing,
                whisper_model_choice,
                prefix_words,
                log_level,
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
