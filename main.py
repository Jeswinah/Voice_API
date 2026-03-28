from fastapi import FastAPI, UploadFile, File
import librosa
import numpy as np
from scipy.spatial.distance import euclidean

app = FastAPI()

# --- Tamil phoneme groups (simplified mapping) ---
TAMIL_PHONEMES = {
    "ழ": "zha",
    "ல": "la",
    "ள": "lla",
    "ர": "ra",
    "ற": "rra",
    "ந": "na",
    "ண": "nna",
}

# --- Extract MFCC features ---
def extract_features(audio_bytes):
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)
    
    y, sr = librosa.load("temp.wav", sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    return np.mean(mfcc.T, axis=0)

# --- Compare audio similarity ---
def compare_audio(f1, f2):
    return euclidean(f1, f2)

# --- Fake phoneme extractor (replace later with real model) ---
def simple_phoneme_estimate(audio_bytes):
    # placeholder logic
    # In real system → wav2vec / whisper phoneme mode
    return ["la", "ra", "zha"]

# --- Tamil letter mapping ---
def map_to_tamil_letters(phonemes):
    reverse_map = {v: k for k, v in TAMIL_PHONEMES.items()}
    return [reverse_map.get(p, "?") for p in phonemes]

# --- Compare phoneme sequences ---
def compare_sequences(ref_seq, user_seq):
    errors = []
    min_len = min(len(ref_seq), len(user_seq))
    
    for i in range(min_len):
        if ref_seq[i] != user_seq[i]:
            errors.append({
                "expected_letter": ref_seq[i],
                "user_letter": user_seq[i],
                "position": i,
                "issue": "phonetic mismatch"
            })
    
    # handle extra/missing
    if len(ref_seq) > len(user_seq):
        for i in range(min_len, len(ref_seq)):
            errors.append({
                "expected_letter": ref_seq[i],
                "user_letter": None,
                "position": i,
                "issue": "missing sound"
            })
    
    if len(user_seq) > len(ref_seq):
        for i in range(min_len, len(user_seq)):
            errors.append({
                "expected_letter": None,
                "user_letter": user_seq[i],
                "position": i,
                "issue": "extra sound"
            })
    
    return errors

# --- MAIN API ---
@app.post("/compare-audio/")
async def compare_audio_api(
    audio_ref: UploadFile = File(...),
    audio_user: UploadFile = File(...)
):
    ref_bytes = await audio_ref.read()
    user_bytes = await audio_user.read()
    
    # Step 1: Feature extraction
    ref_feat = extract_features(ref_bytes)
    user_feat = extract_features(user_bytes)
    
    # Step 2: Sound similarity
    distance = compare_audio(ref_feat, user_feat)
    
    # Step 3: Phoneme estimation
    ref_phonemes = simple_phoneme_estimate(ref_bytes)
    user_phonemes = simple_phoneme_estimate(user_bytes)
    
    # Step 4: Map to Tamil letters
    ref_letters = map_to_tamil_letters(ref_phonemes)
    user_letters = map_to_tamil_letters(user_phonemes)
    
    # Step 5: Compare sequences
    errors = compare_sequences(ref_letters, user_letters)
    
    return {
        "status": "matched" if not errors else "mismatch",
        "distance_score": float(distance),
        "errors": errors
    }